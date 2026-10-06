#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

namespace PluginParamIDs
{
    // Suppressor mode: 0 = Bypass, 1 = Wiener, 2 = RNNoise
    inline const juce::ParameterID mode { "mode", 1 };
}
