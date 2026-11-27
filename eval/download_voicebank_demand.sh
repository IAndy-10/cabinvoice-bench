#!/usr/bin/env bash
# Download and extract the VoiceBank+DEMAND test set.
# Source: Edinburgh DataShare — Valentini-Botinhao et al. (2016)
#
# Usage:
#   ./download_voicebank_demand.sh [output-dir]
#
# Result:
#   <output-dir>/
#     noisy_testset_wav/   ← 824 noisy inputs  (48 kHz)
#     clean_testset_wav/   ← 824 clean refs     (48 kHz)
#     testset_txt/         ← 824 reference transcripts

set -euo pipefail

OUTPUT_DIR="${1:-$(dirname "$0")/data/voicebank_demand}"

NOISY_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"
CLEAN_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"
TXT_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/testset_txt.zip"

mkdir -p "$OUTPUT_DIR"

download_and_extract() {
    local url="$1"
    local dest="$2"
    local filename
    filename="$(basename "$url")"
    local dirname="${filename%.zip}"
    local zip_path="$dest/$filename"

    if [ -d "$dest/$dirname" ]; then
        local count
        count="$(find "$dest/$dirname" -maxdepth 1 \( -name '*.wav' -o -name '*.txt' \) | wc -l | tr -d ' ')"
        echo "Already exists: $dest/$dirname ($count files) — skipping."
        return 0
    fi

    echo "Downloading $filename ..."
    if ! curl -L --retry 3 --fail --progress-bar -o "$zip_path" "$url"; then
        echo "ERROR: Download failed for $url" >&2
        rm -f "$zip_path"
        return 1
    fi

    echo "Extracting $filename ..."
    unzip -q -o "$zip_path" -d "$dest"
    rm "$zip_path"

    local count
    count="$(find "$dest/$dirname" -maxdepth 1 \( -name '*.wav' -o -name '*.txt' \) | wc -l | tr -d ' ')"
    echo "Done: $dest/$dirname ($count files)"
}

download_and_extract "$NOISY_URL" "$OUTPUT_DIR"
download_and_extract "$CLEAN_URL" "$OUTPUT_DIR"
download_and_extract "$TXT_URL"   "$OUTPUT_DIR"

echo ""
echo "VoiceBank+DEMAND test set ready in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  python evaluate.py \\"
echo "    --noisy-dir $OUTPUT_DIR/noisy_testset_wav \\"
echo "    --clean-dir $OUTPUT_DIR/clean_testset_wav \\"
echo "    --enhanced-dirs wiener=out/wiener whisper-tuned=out/whisper_tuned rnnoise=out/rnnoise \\"
echo "    --ref-transcripts $OUTPUT_DIR/testset_txt \\"
echo "    --whisper-model base"
