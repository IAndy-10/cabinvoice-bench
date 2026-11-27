#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

namespace PluginParamIDs
{
    // Suppressor mode: 0 = Bypass, 1 = Wiener, 2 = RNNoise, 3 = Wiener (Whisper-tuned)
    inline const juce::ParameterID mode { "mode", 1 };
    inline const juce::ParameterID inputGain { "inputGain", 1 };
    inline const juce::ParameterID outputGain { "outputGain", 1 };
}
