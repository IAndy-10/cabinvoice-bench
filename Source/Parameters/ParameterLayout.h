#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "ParameterIDs.h"

inline juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add(std::make_unique<juce::AudioParameterChoice>(
        PluginParamIDs::mode,
        "Mode",
        juce::StringArray { "Bypass", "Wiener", "RNNoise" },
        0));

    return layout;
}
