#include "PluginProcessor.h"
#include "../Parameters/ParameterLayout.h"
#include "../Parameters/ParameterIDs.h"
#include "PluginEditor.h"

CleanEarsAudioProcessor::CleanEarsAudioProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input",  juce::AudioChannelSet::mono(), true)
              .withOutput("Output", juce::AudioChannelSet::mono(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    modeParam = apvts.getRawParameterValue(PluginParamIDs::mode.getParamID());
}

CleanEarsAudioProcessor::~CleanEarsAudioProcessor() {}

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

bool CleanEarsAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& mainOut = layouts.getMainOutputChannelSet();
    const auto& mainIn  = layouts.getMainInputChannelSet();

    if (mainOut != juce::AudioChannelSet::mono() &&
        mainOut != juce::AudioChannelSet::stereo())
        return false;

    if (mainOut != mainIn)
        return false;

    return true;
}

void CleanEarsAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    juce::ignoreUnused(sampleRate, samplesPerBlock);
    // TODO: prepare STFT engine, Wiener suppressor, RNNoise wrapper, VAD worker
}

void CleanEarsAudioProcessor::releaseResources() {
    // TODO: stop VAD worker thread, release DSP resources
}

void CleanEarsAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                            juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused(midiMessages);

    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    const auto mode = static_cast<SuppressorMode>(static_cast<int>(modeParam->load()));

    switch (mode)
    {
        case SuppressorMode::Bypass:
            break; // pass-through

        case SuppressorMode::Wiener:
            // TODO: STFT -> Wiener gain -> iSTFT
            break;

        case SuppressorMode::RNNoise:
            // TODO: FIFO -> RNNoise 480-sample frames -> output
            break;
    }
}

bool CleanEarsAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* CleanEarsAudioProcessor::createEditor() {
    return new CleanEarsAudioProcessorEditor(*this);
}

void CleanEarsAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void CleanEarsAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState && xmlState->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new CleanEarsAudioProcessor();
}
