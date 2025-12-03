#!/usr/bin/env python3
"""
Build the VoiceToTextModel with the current architecture and save the untrained
model to the `saved_model` directory in the workspace.

Run inside a Linux environment (or container) to avoid macOS TensorFlow native
threading issues:

  docker run --rm -v "$PWD":/workspace -w /workspace tensorflow/tensorflow:2.12.0 \
      bash -c "python3 save_untrained_model.py"

This will create a `saved_model` folder containing the model (TensorFlow SavedModel format).
"""
import os

MODEL_DIR = "saved_model"

# conservative env in case someone runs locally
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

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

    # Save the untrained model
    try:
        if os.path.exists(MODEL_DIR):
            print(f"Removing existing '{MODEL_DIR}'")
            import shutil
            shutil.rmtree(MODEL_DIR)
        model.save_model(MODEL_DIR)
        print(f"Saved untrained model to '{MODEL_DIR}'")
    except Exception as ex:
        print(f"Failed to save model: {ex}")
        raise
