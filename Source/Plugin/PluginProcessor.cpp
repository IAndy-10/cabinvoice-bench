#include "PluginProcessor.h"
#include "../Parameters/ParameterLayout.h"
#include "../Parameters/ParameterIDs.h"
#include "PluginEditor.h"
#include <cmath>

CabinVoiceBenchAudioProcessor::CabinVoiceBenchAudioProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input",  juce::AudioChannelSet::mono(), true)
              .withOutput("Output", juce::AudioChannelSet::mono(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    modeParam       = apvts.getRawParameterValue(PluginParamIDs::mode.getParamID());
    inputGainParam  = apvts.getRawParameterValue(PluginParamIDs::inputGain.getParamID());
    outputGainParam = apvts.getRawParameterValue(PluginParamIDs::outputGain.getParamID());
    strengthParam   = apvts.getRawParameterValue(PluginParamIDs::strength.getParamID());
}

/// Message thread. First existing candidate wins:
///   1. next to the binary (Standalone: Contents/MacOS; plugins: inside the bundle)
///   2. the bundle's Resources folder
///   3. ~/Library/Application Support/CabinVoice Bench/
///   4. the source tree's models/ (dev builds; set by CMake)
juce::File CabinVoiceBenchAudioProcessor::findVadModel()
{
    const auto exe = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    const juce::File candidates[] = {
        exe.getSiblingFile("silero_vad.onnx"),
        exe.getParentDirectory().getSiblingFile("Resources").getChildFile("silero_vad.onnx"),
        juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
            .getChildFile("CabinVoice Bench").getChildFile("silero_vad.onnx"),
       #ifdef CABINVOICE_BENCH_DEV_MODEL_PATH
        juce::File(CABINVOICE_BENCH_DEV_MODEL_PATH),
       #endif
    };
    for (const auto& f : candidates)
        if (f.existsAsFile())
            return f;
    return {};
}

CabinVoiceBenchAudioProcessor::~CabinVoiceBenchAudioProcessor()
{
    _vad.stop();
}

const juce::String CabinVoiceBenchAudioProcessor::getName() const { return JucePlugin_Name; }
bool CabinVoiceBenchAudioProcessor::acceptsMidi()  const { return false; }
bool CabinVoiceBenchAudioProcessor::producesMidi() const { return false; }
bool CabinVoiceBenchAudioProcessor::isMidiEffect() const { return false; }
double CabinVoiceBenchAudioProcessor::getTailLengthSeconds() const { return 0.0; }
int CabinVoiceBenchAudioProcessor::getNumPrograms() { return 1; }
int CabinVoiceBenchAudioProcessor::getCurrentProgram() { return 0; }
void CabinVoiceBenchAudioProcessor::setCurrentProgram(int) {}
const juce::String CabinVoiceBenchAudioProcessor::getProgramName(int) { return {}; }
void CabinVoiceBenchAudioProcessor::changeProgramName(int, const juce::String&) {}

bool CabinVoiceBenchAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& mainOut = layouts.getMainOutputChannelSet();
    const auto& mainIn  = layouts.getMainInputChannelSet();

    if (mainOut != juce::AudioChannelSet::mono() &&
        mainOut != juce::AudioChannelSet::stereo())
        return false;

    return mainOut == mainIn;
}

void CabinVoiceBenchAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(samplesPerBlock);

    _stft.prepare(sampleRate);

    _wiener.prepare(STFTEngine::kNumBins, { 0.98f, 0.05f, 1.0f });

    _rnnoise.prepare(sampleRate);   // inactive (pass-through) unless sr == 48 kHz
    _rnnoiseAvailable.store(_rnnoise.isActive(), std::memory_order_relaxed);

    const auto modelFile = findVadModel();
    std::string vadError;
    if (!_vad.prepare(sampleRate, modelFile.getFullPathName().toStdString(), vadError))
        DBG("CabinVoice Bench: VAD disabled: " << vadError);   // message thread

    _smoothedInputGain.reset(sampleRate, 0.01);
    _smoothedOutputGain.reset(sampleRate, 0.01);
    _smoothedStrength.reset(sampleRate, 0.05);
    _smoothedStrength.setCurrentAndTargetValue(strengthParam->load());

    _prevMode = SuppressorMode::Bypass;
    setLatencySamples(0);
}

void CabinVoiceBenchAudioProcessor::releaseResources()
{
    _vad.stop();
}

void CabinVoiceBenchAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                            juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused(midiMessages);

    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    const auto numSamples = static_cast<size_t>(buffer.getNumSamples());
    if (numSamples == 0)
        return;

    // Single-mic processor: fold a stereo input to mono in channel 0,
    // process it, then copy the result to every output channel (see end).
    auto* channelData = buffer.getWritePointer(0);
    const int numIn = getTotalNumInputChannels();
    if (numIn > 1)
    {
        for (int ch = 1; ch < numIn; ++ch)
            buffer.addFrom(0, 0, buffer, ch, 0, static_cast<int>(numSamples));
        buffer.applyGain(0, 0, static_cast<int>(numSamples), 1.0f / static_cast<float>(numIn));
    }

    const auto mode = static_cast<SuppressorMode>(static_cast<int>(modeParam->load()));

    // Reset DSP on mode change and update latency
    if (mode != _prevMode)
    {
        _stft.reset();
        _wiener.reset();
        _rnnoise.reset();

        // Defer latency update to message thread via AsyncUpdater
        int newLatency = 0;
        switch (mode)
        {
            case SuppressorMode::Bypass:
                newLatency = 0;
                break;
            case SuppressorMode::Wiener:
                newLatency = static_cast<int>(_stft.getLatencySamples());
                break;
            case SuppressorMode::RNNoise:
                newLatency = static_cast<int>(_rnnoise.getLatencySamples());
                break;
        }
        _pendingLatency.store(newLatency, std::memory_order_relaxed);
        triggerAsyncUpdate();
        _prevMode = mode;
    }

    // Read gain params once per block
    _smoothedInputGain.setTargetValue(
        juce::Decibels::decibelsToGain(inputGainParam->load()));
    _smoothedOutputGain.setTargetValue(
        juce::Decibels::decibelsToGain(outputGainParam->load()));
    _smoothedStrength.setTargetValue(strengthParam->load());
    const float strength = _smoothedStrength.skip(static_cast<int>(numSamples));

    // Apply input gain
    for (size_t i = 0; i < numSamples; ++i)
        channelData[i] *= _smoothedInputGain.getNextValue();

    // Push to VAD (all modes)
    _vad.pushSamples(channelData, numSamples);
    const bool speechActive = _vad.isSpeech();

    // Suppress
    switch (mode)
    {
        case SuppressorMode::Bypass:
            break;

        case SuppressorMode::Wiener:
            _wiener.setStrength(strength);
            _stft.process(channelData, numSamples,
                [this, speechActive](std::complex<float>* bins, size_t numBins) {
                    _wiener.process(bins, numBins, speechActive);
                });
            break;

        case SuppressorMode::RNNoise:
            _rnnoise.process(channelData, numSamples);
            break;
    }

    // Apply output gain + metering
    float sumSq = 0.0f;
    float peak = 0.0f;
    for (size_t i = 0; i < numSamples; ++i)
    {
        channelData[i] *= _smoothedOutputGain.getNextValue();
        const float s = channelData[i];
        sumSq += s * s;
        const float absS = std::abs(s);
        if (absS > peak) peak = absS;
    }

    _rmsLevel.store(std::sqrt(sumSq / static_cast<float>(numSamples)),
                    std::memory_order_relaxed);
    _peakLevel.store(peak, std::memory_order_relaxed);

    for (int ch = 1; ch < getTotalNumOutputChannels(); ++ch)
        buffer.copyFrom(ch, 0, buffer, 0, 0, static_cast<int>(numSamples));
}

bool CabinVoiceBenchAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* CabinVoiceBenchAudioProcessor::createEditor()
{
    return new CabinVoiceBenchAudioProcessorEditor(*this);
}

void CabinVoiceBenchAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void CabinVoiceBenchAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState && xmlState->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
}

void CabinVoiceBenchAudioProcessor::handleAsyncUpdate()
{
    const int latency = _pendingLatency.load(std::memory_order_relaxed);
    if (latency >= 0)
        setLatencySamples(latency);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new CabinVoiceBenchAudioProcessor();
}
