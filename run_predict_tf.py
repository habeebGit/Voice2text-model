#!/usr/bin/env python3
"""
Run model.predict using TensorFlow inside a safe (Linux) environment.
This script sets conservative threading env vars before importing TensorFlow
and then builds the VoiceToTextModel and runs predict on `test_audio.wav`.

Note: The model will be untrained unless you have saved weights in `saved_model`.
"""
import os
# limit native threads before TF import
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

AUDIO_FILE = "test_audio.wav"
MODEL_PATH = "saved_model"

try:
    from voice2text import VoiceToTextModel
except Exception as e:
    print(f"Failed to import VoiceToTextModel: {e}")
    raise

if __name__ == '__main__':
    model = VoiceToTextModel(num_features=80)
    demo_texts = ['hello world', 'machine learning', 'speech recognition']
    vocab_size = model.create_vocabulary(demo_texts)
    model.build_model(vocab_size)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_model(MODEL_PATH)
            print(f"Loaded trained model from '{MODEL_PATH}'")
        except Exception as e:
            print(f"Failed to load model from '{MODEL_PATH}': {e}\nProceeding with untrained model.")

    if not os.path.exists(AUDIO_FILE):
        print(f"Audio file '{AUDIO_FILE}' not found. Generate with generate_input_audio.py")
    else:
        try:
            pred = model.predict(AUDIO_FILE)
            print(f"Predicted text: {pred}")
        except Exception as e:
            print(f"Prediction failed: {e}")
            raise
