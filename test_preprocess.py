#!/usr/bin/env python3
"""
Quick tester that runs the same preprocessing used by VoiceToTextModel
on the generated WAV files and prints shapes and basic stats.
"""
import numpy as np
import librosa

AUDIO_FILES = ["audio1.wav", "audio2.wav", "audio3.wav", "test_audio.wav"]


def preprocess_audio(audio_path, sample_rate=16000, num_features=80):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
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
    for f in AUDIO_FILES:
        try:
            feats = preprocess_audio(f)
            print(f"{f}: features shape={feats.shape}, dtype={feats.dtype}, min={feats.min():.4f}, max={feats.max():.4f}")
        except Exception as e:
            print(f"{f}: ERROR -> {e}")
