#include "PluginEditor.h"
#include "../Parameters/ParameterIDs.h"

CabinVoiceBenchAudioProcessorEditor::CabinVoiceBenchAudioProcessorEditor(CabinVoiceBenchAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    modeSelector.addItemList({ "Bypass", "Wiener", "RNNoise" }, 1);
    modeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        processorRef.apvts, PluginParamIDs::mode.getParamID(), modeSelector);
    addAndMakeVisible(modeSelector);

    strengthSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    strengthSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    strengthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processorRef.apvts, PluginParamIDs::strength.getParamID(), strengthSlider);
    strengthLabel.setText("Wiener strength", juce::dontSendNotification);
    strengthLabel.attachToComponent(&strengthSlider, true);
    addAndMakeVisible(strengthSlider);

    setSize(420, 210);
    startTimerHz(30);
}

void CabinVoiceBenchAudioProcessorEditor::timerCallback()
{
    vadProbability = processorRef.getVadProbability();
    speech = processorRef.isSpeechActive();
    level = processorRef.getRmsLevel();
    repaint(getLocalBounds().removeFromBottom(70));
}

void CabinVoiceBenchAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1e1f22));
    g.setColour(juce::Colours::white);
    g.setFont(18.0f);
    g.drawText("CabinVoice Bench", getLocalBounds().removeFromTop(40), juce::Justification::centred);

    auto status = getLocalBounds().reduced(20).removeFromBottom(60);

    // VAD probability bar + speech indicator
    auto row = status.removeFromTop(22);
    g.setFont(13.0f);
    g.setColour(juce::Colours::lightgrey);
    g.drawText("VAD", row.removeFromLeft(40), juce::Justification::centredLeft);
    auto dot = row.removeFromRight(22).reduced(5);
    g.setColour(speech ? juce::Colour(0xff3ccf6e) : juce::Colour(0xff4a4c52));
    g.fillEllipse(dot.toFloat());
    auto bar = row.reduced(4, 6);
    g.setColour(juce::Colour(0xff34363b));
    g.fillRect(bar);
    g.setColour(juce::Colour(0xff6aa9ff));
    g.fillRect(bar.withWidth(juce::roundToInt(bar.getWidth() * juce::jlimit(0.0f, 1.0f, vadProbability))));

    // Status line: why something might be off, in plain words
    juce::String msg;
    if (!processorRef.isVadModelLoaded())
        msg << "VAD off: silero_vad.onnx not found. ";
    if (!processorRef.isRNNoiseAvailable())
        msg << "RNNoise needs 48 kHz (passing through). ";
    if (msg.isEmpty())
        msg << "Output " << juce::String(juce::Decibels::gainToDecibels(level, -90.0f), 1) << " dBFS RMS";
    g.setColour(juce::Colours::grey);
    g.drawText(msg, status.removeFromTop(24), juce::Justification::centredLeft);
}

void CabinVoiceBenchAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced(20);
    area.removeFromTop(30);
    modeSelector.setBounds(area.removeFromTop(28));
    area.removeFromTop(12);
    strengthSlider.setBounds(area.removeFromTop(26).withTrimmedLeft(110));
}
