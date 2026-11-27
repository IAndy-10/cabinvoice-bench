cmake_minimum_required(VERSION 3.22)

project(CleanEars VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# JUCE - FetchContent
# ============================================================

include(FetchContent)

FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(JUCE)

# ============================================================
# ONNX Runtime
# ============================================================

# The microsoft/onnxruntime GitHub release URLs no longer resolve (404 as of
# 2026-11), so ORT is taken from a local install instead of FetchContent.
#   macOS:  brew install onnxruntime
#   other:  extract an ORT package and pass -DONNXRUNTIME_ROOT=<dir>
# <dir> must contain include/ (or include/onnxruntime/) and lib/.

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime install prefix")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT} /opt/homebrew/opt/onnxruntime /usr/local/opt/onnxruntime /usr/local /usr
    PATH_SUFFIXES include/onnxruntime include
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_ROOT} /opt/homebrew/opt/onnxruntime /usr/local/opt/onnxruntime /usr/local /usr
    PATH_SUFFIXES lib lib64
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime not found. macOS: `brew install onnxruntime`. "
        "Otherwise set -DONNXRUNTIME_ROOT=<prefix>.")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

# ============================================================
# RNNoise
# ============================================================

FetchContent_Declare(
    rnnoise
    GIT_REPOSITORY https://github.com/xiph/rnnoise.git
    GIT_TAG v0.1.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(rnnoise)

# Build RNNoise as a static library from sources (v0.2 branch has simpler layout)
add_library(rnnoise_lib STATIC
    "${rnnoise_SOURCE_DIR}/src/denoise.c"
    "${rnnoise_SOURCE_DIR}/src/rnn.c"
    "${rnnoise_SOURCE_DIR}/src/rnn_data.c"
    "${rnnoise_SOURCE_DIR}/src/kiss_fft.c"
    "${rnnoise_SOURCE_DIR}/src/pitch.c"
    "${rnnoise_SOURCE_DIR}/src/celt_lpc.c"
)
target_include_directories(rnnoise_lib
    PUBLIC "${rnnoise_SOURCE_DIR}/include"
    PRIVATE "${rnnoise_SOURCE_DIR}/src"
)

# ============================================================
# Plugin Definition
# ============================================================

juce_add_plugin(CleanEars
    COMPANY_NAME "Italo Rojas"
    BUNDLE_ID "com.italorojas.CleanEars"
    PLUGIN_MANUFACTURER_CODE Itro
    PLUGIN_CODE Cler
    FORMATS Standalone VST3 AU
    PRODUCT_NAME "Clean Ears"

    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE

    COPY_PLUGIN_AFTER_BUILD TRUE

    VST3_CATEGORIES Fx Tools
    AU_MAIN_TYPE kAudioUnitType_Effect
)

juce_generate_juce_header(CleanEars)

# ============================================================
# Source Files
# ============================================================

target_sources(CleanEars PRIVATE
    Source/Plugin/PluginProcessor.cpp
    Source/Plugin/PluginProcessor.h
    Source/Plugin/PluginEditor.cpp
    Source/Plugin/PluginEditor.h

    Source/Parameters/ParameterIDs.h
    Source/Parameters/ParameterLayout.h

    # DSP infrastructure (header-only)
    Source/DSP/Common/SPSCRingBuffer.h
    Source/DSP/Common/FIFOAdapter.h
    Source/DSP/STFT/STFTEngine.h
    Source/DSP/Wiener/WienerSuppressor.h
    Source/DSP/RNNoise/RNNoiseWrapper.h
    Source/DSP/RNNoise/RNNoiseWrapper.cpp
    Source/DSP/VAD/SileroModel.h
    Source/DSP/VAD/SileroModel.cpp
    Source/DSP/VAD/SileroVAD.h
    Source/DSP/VAD/SileroVAD.cpp
    Source/DSP/VAD/Downsampler.h
)

target_include_directories(CleanEars PRIVATE
    Source/Plugin
    Source/DSP
    Source/Parameters
)

# ============================================================
# Compile Definitions & Linking
# ============================================================

target_compile_definitions(CleanEars PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
)

# Dev builds find the VAD model in the source tree; shipped builds look next to
# the binary or in ~/Library/Application Support/Clean Ears/ (see findVadModel).
target_compile_definitions(CleanEars PRIVATE
    CLEAN_EARS_DEV_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/silero_vad.onnx"
)

if(APPLE)
    target_compile_options(CleanEars PRIVATE
        -Wno-deprecated-declarations
    )
endif()

target_link_libraries(CleanEars PRIVATE
    onnxruntime
    rnnoise_lib
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_dsp
    juce::juce_gui_basics
PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# ============================================================
# CLI Block Processor (offline evaluation tool)
# ============================================================

juce_add_console_app(CleanEarsCLI
    PRODUCT_NAME "clean-ears-cli"
    COMPANY_NAME "Italo Rojas"
)

target_sources(CleanEarsCLI PRIVATE
    Source/CLI/BlockProcessor.cpp
    Source/DSP/RNNoise/RNNoiseWrapper.cpp
    Source/DSP/VAD/SileroModel.cpp
)

target_include_directories(CleanEarsCLI PRIVATE
    Source/DSP
    Source/Parameters
)

target_compile_definitions(CleanEarsCLI PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
)

target_link_libraries(CleanEarsCLI PRIVATE
    onnxruntime
    rnnoise_lib
    juce::juce_audio_formats
    juce::juce_dsp
    juce::juce_core
    juce::juce_recommended_config_flags
)

# ============================================================
# Tests  (run all with: ctest --test-dir build --output-on-failure)
# ============================================================

enable_testing()

add_executable(TestSTFT Tests/TestSTFTPerfectReconstruction.cpp)
target_include_directories(TestSTFT PRIVATE Source/DSP)
target_link_libraries(TestSTFT PRIVATE juce::juce_dsp)
add_test(NAME STFTPerfectReconstruction COMMAND TestSTFT)

# Measures each mode's end-to-end delay and checks it equals the reported latency.
add_executable(TestLatency
    Tests/TestLatency.cpp
    Source/DSP/RNNoise/RNNoiseWrapper.cpp
)
target_include_directories(TestLatency PRIVATE Source/DSP)
target_link_libraries(TestLatency PRIVATE rnnoise_lib juce::juce_dsp)
add_test(NAME ModeLatency COMMAND TestLatency)

# Loads the shipped Silero v5 model and checks it runs and separates silence from tone/noise.
add_executable(TestSileroVAD
    Tests/TestSileroVAD.cpp
    Source/DSP/VAD/SileroModel.cpp
)
target_include_directories(TestSileroVAD PRIVATE Source/DSP)
target_link_libraries(TestSileroVAD PRIVATE onnxruntime)
add_test(NAME SileroVADModel
    COMMAND TestSileroVAD "${CMAKE_CURRENT_SOURCE_DIR}/models/silero_vad.onnx")
