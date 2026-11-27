#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "../DSP/STFT/STFTEngine.h"
#include "../DSP/Wiener/WienerSuppressor.h"
#include "../DSP/RNNoise/RNNoiseWrapper.h"
#include "../DSP/VAD/SileroVAD.h"
#include <atomic>

class CabinVoiceBenchAudioProcessor final : public juce::AudioProcessor,
                                      private juce::AsyncUpdater {
public:
    enum class SuppressorMode {
        Bypass = 0,
        Wiener,
        RNNoise
    };

    CabinVoiceBenchAudioProcessor();
    ~CabinVoiceBenchAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    float getRmsLevel() const { return _rmsLevel.load(std::memory_order_relaxed); }
    float getPeakLevel() const { return _peakLevel.load(std::memory_order_relaxed); }
    float getVadProbability() const { return _vad.getProbability(); }
    bool isSpeechActive() const { return _vad.isSpeech(); }
    bool isVadModelLoaded() const { return _vad.isModelLoaded(); }
    bool isRNNoiseAvailable() const { return _rnnoiseAvailable.load(std::memory_order_relaxed); }

    juce::AudioProcessorValueTreeState apvts;

private:
    std::atomic<float>* modeParam = nullptr;
    std::atomic<float>* inputGainParam = nullptr;
    std::atomic<float>* outputGainParam = nullptr;
    std::atomic<float>* strengthParam = nullptr;

    STFTEngine       _stft;
    WienerSuppressor _wiener;
    RNNoiseWrapper   _rnnoise;
    SileroVAD        _vad;

    juce::SmoothedValue<float> _smoothedInputGain;
    juce::SmoothedValue<float> _smoothedOutputGain;
    juce::SmoothedValue<float> _smoothedStrength;

    std::atomic<bool> _rnnoiseAvailable{ false };

    static juce::File findVadModel();

    SuppressorMode _prevMode = SuppressorMode::Bypass;

    std::atomic<float> _rmsLevel{0.0f};
    std::atomic<float> _peakLevel{0.0f};

    // Deferred latency update (setLatencySamples not RT-safe)
    std::atomic<int> _pendingLatency{-1};
    void handleAsyncUpdate() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CabinVoiceBenchAudioProcessor)
};
