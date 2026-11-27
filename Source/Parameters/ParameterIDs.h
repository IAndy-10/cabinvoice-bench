#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

namespace PluginParamIDs
{
    // Suppressor mode: 0 = Bypass, 1 = Wiener, 2 = RNNoise
    // (v2: the "Wiener (Whisper-tuned)" preset was replaced by `strength`)
    inline const juce::ParameterID mode { "mode", 2 };
    // Wiener suppression strength: applied gain = G^strength, 0 = none, 1 = full
    inline const juce::ParameterID strength { "strength", 1 };
    inline const juce::ParameterID inputGain { "inputGain", 1 };
    inline const juce::ParameterID outputGain { "outputGain", 1 };
}
