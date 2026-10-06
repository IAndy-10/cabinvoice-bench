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

set(ONNXRUNTIME_VERSION "1.17.1")

if(APPLE)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
elseif(WIN32)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
endif()

FetchContent_Declare(
    onnxruntime
    URL ${ONNXRUNTIME_URL}
)

FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}onnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

if(APPLE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.dylib"
    )
endif()

# ============================================================
# Plugin Definition
# ============================================================

juce_add_plugin(CleanEars
    COMPANY_NAME "Italo Rojas"
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

if(APPLE)
    target_compile_options(CleanEars PRIVATE
        -Wno-deprecated-declarations
    )
endif()

target_link_libraries(CleanEars PRIVATE
    onnxruntime
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_gui_basics
PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)
