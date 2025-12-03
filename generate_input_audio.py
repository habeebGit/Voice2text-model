#!/usr/bin/env python3
"""
Generate simple synthetic, speech-like WAV files for testing the VoiceToTextModel.
These files are 16 kHz, mono, PCM16 and compatible with the model's
`preprocess_audio(..., sample_rate=16000)` function.

Run:
    python3 generate_input_audio.py

This will create: audio1.wav, audio2.wav, audio3.wav, test_audio.wav
"""
import numpy as np
import soundfile as sf

def make_speech_like(duration, sr=16000, freqs=(120, 220)):
    """Create a short, speech-like waveform using multiple sinusoids
    with an amplitude envelope to mimic syllable bursts and low-level noise.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # amplitude envelope with random bursts (syllable-like)
    envelope = np.zeros_like(t)
    n_bursts = max(3, int(duration * 3))
    for i in range(n_bursts):
        start = np.random.uniform(0, max(0.0, duration - 0.15))
        dur = np.random.uniform(0.06, 0.25)
        s = int(start * sr)
        e = min(len(t), s + int(dur * sr))
        if e > s:
            envelope[s:e] += np.hanning(e - s)
    if envelope.max() > 0:
        envelope = envelope / envelope.max()

    # sum of sinusoids with slight frequency modulation
    signal = np.zeros_like(t)
    for f in freqs:
        fm = f + 5.0 * np.sin(2 * np.pi * 0.5 * t + np.random.randn())
        signal += np.sin(2 * np.pi * fm * t + np.random.uniform(0, 2 * np.pi))

    signal = signal * envelope

    # low-level breath/room noise
    noise = 0.01 * np.random.normal(0, 1, len(t))
    signal = signal + noise

    # normalize to avoid clipping
    maxval = np.max(np.abs(signal))
    if maxval > 0:
        signal = 0.95 * signal / maxval

    return signal.astype(np.float32)

if __name__ == "__main__":
    sr = 16000
    specs = [
        (2.5, (120, 200)),
        (3.0, (150, 300)),
        (2.0, (100, 250)),
    ]

    for i, (dur, freqs) in enumerate(specs, start=1):
        wav = make_speech_like(dur, sr=sr, freqs=freqs)
        filename = f"audio{i}.wav"
        sf.write(filename, wav, sr, subtype="PCM_16")
        print(f"Wrote {filename} ({dur}s, sr={sr})")

    test = make_speech_like(4.0, sr=sr, freqs=(130, 260))
    sf.write("test_audio.wav", test, sr, subtype="PCM_16")
    print("Wrote test_audio.wav (4.0s, sr=16000)")
