#pragma once

#include "PluginProcessor.h"

class CleanEarsAudioProcessorEditor final : public juce::AudioProcessorEditor {
public:
    explicit CleanEarsAudioProcessorEditor(CleanEarsAudioProcessor&);
    ~CleanEarsAudioProcessorEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    CleanEarsAudioProcessor& processorRef;

    juce::ComboBox modeSelector;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> modeAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CleanEarsAudioProcessorEditor)
};
