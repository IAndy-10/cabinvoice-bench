#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "ParameterIDs.h"

inline juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add(std::make_unique<juce::AudioParameterChoice>(
        PluginParamIDs::mode,
        "Mode",
        juce::StringArray { "Bypass", "Wiener", "RNNoise", "Wiener (Whisper-tuned)" },
        0));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        PluginParamIDs::inputGain,
        "Input Gain",
        juce::NormalisableRange<float>(-24.0f, 12.0f, 0.1f),
        0.0f));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        PluginParamIDs::outputGain,
        "Output Gain",
        juce::NormalisableRange<float>(-24.0f, 12.0f, 0.1f),
        0.0f));

    return layout;
}
