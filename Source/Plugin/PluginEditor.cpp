#include "PluginEditor.h"
#include "../Parameters/ParameterIDs.h"

CleanEarsAudioProcessorEditor::CleanEarsAudioProcessorEditor(CleanEarsAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    modeSelector.addItemList({ "Bypass", "Wiener", "RNNoise" }, 1);
    modeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        processorRef.apvts, PluginParamIDs::mode.getParamID(), modeSelector);

    addAndMakeVisible(modeSelector);
    setSize(300, 120);
}

void CleanEarsAudioProcessorEditor::paint(juce::Graphics& g) {
    g.fillAll(juce::Colours::darkgrey);
    g.setColour(juce::Colours::white);
    g.setFont(18.0f);
    g.drawText("Clean Ears", getLocalBounds().removeFromTop(40), juce::Justification::centred);
}

void CleanEarsAudioProcessorEditor::resized() {
    auto area = getLocalBounds().reduced(20);
    area.removeFromTop(40);
    modeSelector.setBounds(area.removeFromTop(30));
}
