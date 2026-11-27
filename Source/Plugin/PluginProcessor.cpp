#include "PluginProcessor.h"
#include "../Parameters/ParameterLayout.h"
#include "../Parameters/ParameterIDs.h"
#include "PluginEditor.h"
#include <cmath>

CleanEarsAudioProcessor::CleanEarsAudioProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input",  juce::AudioChannelSet::mono(), true)
              .withOutput("Output", juce::AudioChannelSet::mono(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    modeParam       = apvts.getRawParameterValue(PluginParamIDs::mode.getParamID());
    inputGainParam  = apvts.getRawParameterValue(PluginParamIDs::inputGain.getParamID());
    outputGainParam = apvts.getRawParameterValue(PluginParamIDs::outputGain.getParamID());
}

CleanEarsAudioProcessor::~CleanEarsAudioProcessor()
{
    _vad.stop();
}

const juce::String CleanEarsAudioProcessor::getName() const { return JucePlugin_Name; }
bool CleanEarsAudioProcessor::acceptsMidi()  const { return false; }
bool CleanEarsAudioProcessor::producesMidi() const { return false; }
bool CleanEarsAudioProcessor::isMidiEffect() const { return false; }
double CleanEarsAudioProcessor::getTailLengthSeconds() const { return 0.0; }
int CleanEarsAudioProcessor::getNumPrograms() { return 1; }
int CleanEarsAudioProcessor::getCurrentProgram() { return 0; }
void CleanEarsAudioProcessor::setCurrentProgram(int) {}
const juce::String CleanEarsAudioProcessor::getProgramName(int) { return {}; }
void CleanEarsAudioProcessor::changeProgramName(int, const juce::String&) {}

bool CleanEarsAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& mainOut = layouts.getMainOutputChannelSet();
    const auto& mainIn  = layouts.getMainInputChannelSet();

    if (mainOut != juce::AudioChannelSet::mono() &&
        mainOut != juce::AudioChannelSet::stereo())
        return false;

    return mainOut == mainIn;
}

void CleanEarsAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(samplesPerBlock);

    _stft.prepare(sampleRate);

    // Standard Wiener: alpha=0.98, gainFloor=0.05 (-26 dB)
    _wiener.prepare(STFTEngine::kNumBins, {0.98f, 0.05f});

    // Whisper-tuned Wiener: less aggressive, preserves speech detail for ASR
    _wienerWhisperTuned.prepare(STFTEngine::kNumBins, {0.92f, 0.10f});

    _rnnoise.prepare(sampleRate);

    // Look for silero_vad.onnx next to the executable
    const auto modelPath = juce::File::getSpecialLocation(
        juce::File::currentExecutableFile)
        .getParentDirectory()
        .getChildFile("silero_vad.onnx")
        .getFullPathName()
        .toStdString();

    _vad.prepare(sampleRate,
                 juce::File(modelPath).existsAsFile() ? modelPath : "");

    _smoothedInputGain.reset(sampleRate, 0.01);
    _smoothedOutputGain.reset(sampleRate, 0.01);

    _prevMode = SuppressorMode::Bypass;
    setLatencySamples(0);
}

void CleanEarsAudioProcessor::releaseResources()
{
    _vad.stop();
}

void CleanEarsAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                            juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused(midiMessages);

    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    const auto numSamples = static_cast<size_t>(buffer.getNumSamples());
    auto* channelData = buffer.getWritePointer(0);

    const auto mode = static_cast<SuppressorMode>(static_cast<int>(modeParam->load()));

    // Reset DSP on mode change and update latency
    if (mode != _prevMode)
    {
        _stft.reset();
        _wiener.reset();
        _wienerWhisperTuned.reset();
        _rnnoise.reset();

        // Defer latency update to message thread via AsyncUpdater
        int newLatency = 0;
        switch (mode)
        {
            case SuppressorMode::Bypass:
                newLatency = 0;
                break;
            case SuppressorMode::Wiener:
            case SuppressorMode::WienerWhisperTuned:
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
            _stft.process(channelData, numSamples,
                [this, speechActive](std::complex<float>* bins, size_t numBins) {
                    _wiener.process(bins, numBins, speechActive);
                });
            break;

        case SuppressorMode::WienerWhisperTuned:
            _stft.process(channelData, numSamples,
                [this, speechActive](std::complex<float>* bins, size_t numBins) {
                    _wienerWhisperTuned.process(bins, numBins, speechActive);
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
}

bool CleanEarsAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* CleanEarsAudioProcessor::createEditor()
{
    return new CleanEarsAudioProcessorEditor(*this);
}

void CleanEarsAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void CleanEarsAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState && xmlState->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
}

void CleanEarsAudioProcessor::handleAsyncUpdate()
{
    const int latency = _pendingLatency.load(std::memory_order_relaxed);
    if (latency >= 0)
        setLatencySamples(latency);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new CleanEarsAudioProcessor();
}
