#pragma once

#include "PluginProcessor.h"

class CleanEarsAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                            private juce::Timer {
public:
    explicit CleanEarsAudioProcessorEditor(CleanEarsAudioProcessor&);
    ~CleanEarsAudioProcessorEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    CleanEarsAudioProcessor& processorRef;

    juce::ComboBox modeSelector;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> modeAttachment;

    juce::Slider strengthSlider;
    juce::Label strengthLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> strengthAttachment;

    // Polled from atomics on the message thread at 30 Hz.
    float vadProbability = 0.0f;
    bool speech = false;
    float level = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CleanEarsAudioProcessorEditor)
};
