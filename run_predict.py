#!/usr/bin/env python3
"""
Safe preprocessing-only predictor.
This script does not import TensorFlow or the VoiceToTextModel. It simply
runs the same preprocessing pipeline on `test_audio.wav` and prints feature
shape and statistics so you can verify the input pipeline works on the
generated WAV files.

Run:
    python3 run_predict.py
"""
import os
import sys

AUDIO_FILE = "test_audio.wav"

try:
    import numpy as np
    import librosa
except Exception as e:
    print(f"Required packages missing: {e}")
    sys.exit(1)


def preprocess_audio(audio_path, sample_rate=16000, num_features=80):
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=num_features,
        fmax=8000
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)
    return log_mel_spec.T  # (time, features)


if __name__ == '__main__':
    if not os.path.exists(AUDIO_FILE):
        print(f"Audio file '{AUDIO_FILE}' not found. Generate files with generate_input_audio.py")
        sys.exit(1)

    try:
        feats = preprocess_audio(AUDIO_FILE)
        print(f"Preprocessing OK for '{AUDIO_FILE}': shape={feats.shape}, dtype={feats.dtype}, min={feats.min():.4f}, max={feats.max():.4f}")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        sys.exit(1)
