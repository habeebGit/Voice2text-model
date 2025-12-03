#!/usr/bin/env python3
"""
Minimal CTC-compatible training script.

Instructions: run inside the TensorFlow Docker image to avoid macOS TF native issues:

docker run --rm -v "$PWD":/workspace -w /workspace tensorflow/tensorflow:2.12.0 \
  bash -c "pip install --no-cache-dir librosa soundfile && python3 train_ctc.py"

This script:
- Loads the synthetic WAV files (audio1.wav..audio3.wav) and their demo transcriptions
- Builds the VoiceToTextModel and the character vocabulary
- Runs a tiny training loop using tf.GradientTape and keras.backend.ctc_batch_cost
- Saves the trained model to `trained_model/`

Note: This is a minimal demo for pipeline correctness, not for real ASR quality.
"""
import os
import numpy as np

# Conservative env
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

try:
    from voice2text import VoiceToTextModel
except Exception as e:
    print(f"Failed to import VoiceToTextModel: {e}")
    raise

import tensorflow as tf
from tensorflow import keras

# Training data (synthetic files and simple transcriptions)
AUDIO_FILES = ["audio1.wav", "audio2.wav", "audio3.wav"]
TRANSCRIPTIONS = ["hello world", "machine learning", "speech recognition"]

TRAINED_DIR = "trained_model"
EPOCHS = 6
LR = 1e-4


def load_features(model_obj, path):
    feats = model_obj.preprocess_audio(path)
    return feats.astype(np.float32)


if __name__ == '__main__':
    # Build model and vocabulary
    model_obj = VoiceToTextModel(num_features=80)
    vocab_size = model_obj.create_vocabulary(TRANSCRIPTIONS)
    model_obj.build_model(vocab_size)

    # Prepare data (features and encoded labels)
    X = [load_features(model_obj, p) for p in AUDIO_FILES]
    y = [model_obj.encode_text(t) for t in TRANSCRIPTIONS]

    # Pad labels to max length (padding value 0, label_length gives true length)
    label_lengths = [len(seq) for seq in y]
    max_label_len = max(label_lengths)
    y_padded = np.zeros((len(y), max_label_len), dtype=np.int32)
    for i, seq in enumerate(y):
        y_padded[i, :len(seq)] = seq

    # Shift labels by +1 so that 0 is reserved for padding (CTC dense-to-sparse helper expects this)
    y_padded = y_padded + 1
    # Now valid character indices are in [1..vocab_size]

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=LR)

    # Simple training loop (per-sample gradient step)
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for i in range(len(X)):
            x = X[i]
            label = y_padded[i:i+1]
            label_len = np.array([label_lengths[i]], dtype=np.int32)

            with tf.GradientTape() as tape:
                # model expects batch dim
                x_in = tf.convert_to_tensor(np.expand_dims(x, axis=0))  # (1, T, features)
                y_pred = model_obj.model(x_in, training=True)  # (1, T_out, vocab+1)

                # compute input_length from prediction time dimension
                input_len = np.array([y_pred.shape[1]], dtype=np.int32)

                # Ensure there are enough time-steps for CTC given the label length.
                # CTC requires at least (2 * label_len + 1) timesteps to represent labels
                # (accounting for blanks and repeats). If not enough, pad y_pred along
                # the time axis by repeating the last frame (low-cost hack for demo).
                required_timesteps = 2 * int(label_len) + 1
                current_timesteps = int(y_pred.shape[1])
                if current_timesteps < required_timesteps:
                    pad = required_timesteps - current_timesteps
                    # Pad along time axis (second dim) with the last timestep's logits
                    last_frame = y_pred[:, -1:, :]
                    pads = tf.repeat(last_frame, repeats=pad, axis=1)
                    y_pred = tf.concat([y_pred, pads], axis=1)
                    input_len = np.array([y_pred.shape[1]], dtype=np.int32)

                # Convert dense padded labels (0 used for padding) to a SparseTensor
                # required by ctc_batch_cost. Use original label lengths (before padding).
                label_tensor = tf.convert_to_tensor(label, dtype=tf.int32)  # shape (batch, max_label_len)
                # Build SparseTensor from dense padded labels (0 is padding)
                nonzero_positions = tf.where(tf.not_equal(label_tensor, 0))  # (N, 2) indices
                nonzero_values = tf.gather_nd(label_tensor, nonzero_positions)
                dense_shape = tf.cast(tf.shape(label_tensor), dtype=tf.int64)
                labels_sparse = tf.sparse.SparseTensor(indices=tf.cast(nonzero_positions, tf.int64),
                                                      values=tf.cast(nonzero_values, tf.int32),
                                                      dense_shape=dense_shape)

                # Use tf.nn.ctc_loss which accepts SparseTensor labels directly.
                # Convert softmax outputs to log-probabilities for the loss.
                log_probs = tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0))

                # blank index: last class (vocab_size)
                blank_index = tf.cast(tf.shape(y_pred)[-1] - 1, tf.int32)

                loss_tensor = tf.nn.ctc_loss(
                    labels=labels_sparse,
                    logits=log_probs,
                    label_length=tf.convert_to_tensor(label_len, dtype=tf.int32),
                    logit_length=tf.convert_to_tensor(input_len, dtype=tf.int32),
                    logits_time_major=False,
                    blank_index=blank_index,
                )
                # loss_tensor shape is (batch,) -> reduce to scalar
                loss = tf.reduce_mean(loss_tensor, axis=None)

            grads = tape.gradient(loss, model_obj.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_obj.model.trainable_variables))

            total_loss += float(loss.numpy())

        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch}/{EPOCHS} - avg_loss={avg_loss:.6f}")

    # Save trained model
    try:
        if os.path.exists(TRAINED_DIR):
            import shutil
            shutil.rmtree(TRAINED_DIR)
        model_obj.save_model(TRAINED_DIR)
        print(f"Saved trained model to '{TRAINED_DIR}'")
    except Exception as e:
        print(f"Failed to save trained model: {e}")
        raise
