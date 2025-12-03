import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import librosa
import soundfile as sf

class VoiceToTextModel:
    """
    A deep learning model for converting speech audio to text using TensorFlow.
    Uses a CNN + Bidirectional LSTM architecture with CTC loss.
    """
    
    def __init__(self, num_features=80, max_text_length=200):
        self.num_features = num_features  # Number of mel-frequency features
        self.max_text_length = max_text_length
        self.model = None
        self.char_to_num = None
        self.num_to_char = None
        
    def build_model(self, vocab_size):
        """Build the voice-to-text model architecture."""
        
        # Input layer
        input_spectrogram = layers.Input(
            shape=(None, self.num_features), 
            name="input"
        )
        
        # Expand dimensions for CNN
        x = layers.Reshape((-1, self.num_features, 1))(input_spectrogram)
        
        # CNN layers for feature extraction
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Reshape for RNN
        # Flatten spatial dims (width, channels) for each time step so RNN
        # receives a sequence of feature vectors. Using TimeDistributed(Flatten)
        # keeps the graph symbolic-safe (no dynamic tf.shape usage here).
        x = layers.TimeDistributed(layers.Flatten())(x)

        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.2)
        )(x)
        x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.2)
        )(x)
        
        # Dense layer for character prediction
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer (vocab_size + 1 for CTC blank token)
        output = layers.Dense(vocab_size + 1, activation="softmax")(x)
        
        self.model = keras.Model(input_spectrogram, output, name="voice_to_text")
        return self.model
    
    def preprocess_audio(self, audio_path, sample_rate=16000):
        """
        Load and preprocess audio file to mel-spectrogram features.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            
        Returns:
            Mel-spectrogram features
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=self.num_features,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)
        
        return log_mel_spec.T  # Transpose to (time, features)
    
    def create_vocabulary(self, texts):
        """Create character-to-number and number-to-character mappings."""
        chars = sorted(set(''.join(texts)))
        self.char_to_num = {char: idx for idx, char in enumerate(chars)}
        self.num_to_char = {idx: char for idx, char in enumerate(chars)}
        return len(chars)
    
    def encode_text(self, text):
        """Encode text to numerical sequence."""
        return [self.char_to_num[char] for char in text if char in self.char_to_num]
    
    def decode_prediction(self, prediction):
        """Decode model prediction to text using greedy decoding."""
        # Get the most likely character at each time step
        input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
        
        # Use greedy decoding
        results = keras.backend.ctc_decode(
            prediction, 
            input_length=input_len, 
            greedy=True
        )[0][0]
        
        # Convert to text
        output_text = []
        for result in results:
            text = ''.join([self.num_to_char[int(idx)] for idx in result if int(idx) != -1])
            output_text.append(text)
        
        return output_text
    
    def compile_model(self):
        """Compile the model with CTC loss."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=self.ctc_loss
        )
    
    @staticmethod
    def ctc_loss(y_true, y_pred):
        """CTC (Connectionist Temporal Classification) loss function."""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        
        input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len,), dtype="int64")
        
        loss = keras.backend.ctc_batch_cost(
            y_true, y_pred, input_length, label_length
        )
        return loss
    
    def train(self, audio_paths, transcriptions, epochs=50, batch_size=32):
        """
        Train the model on audio files and their transcriptions.
        
        Args:
            audio_paths: List of paths to audio files
            transcriptions: List of corresponding text transcriptions
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Create vocabulary
        vocab_size = self.create_vocabulary(transcriptions)
        
        # Build and compile model
        self.build_model(vocab_size)
        self.compile_model()
        
        # Preprocess data
        X = [self.preprocess_audio(path) for path in audio_paths]
        y = [self.encode_text(text) for text in transcriptions]
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            lambda: zip(X, y),
            output_signature=(
                tf.TensorSpec(shape=(None, self.num_features), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        
        dataset = dataset.padded_batch(batch_size)
        
        # Train
        history = self.model.fit(
            dataset,
            epochs=epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        
        return history
    
    def predict(self, audio_path):
        """
        Predict text from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Predicted text
        """
        # Preprocess audio
        features = self.preprocess_audio(audio_path)
        features = np.expand_dims(features, axis=0)
        
        # Predict
        prediction = self.model.predict(features)
        
        # Decode
        text = self.decode_prediction(prediction)
        
        return text[0]
    
    def save_model(self, path):
        """Save the model to disk."""
        self.model.save(path)
        
    def load_model(self, path):
        """Load a saved model from disk."""
        self.model = keras.models.load_model(
            path, 
            custom_objects={'ctc_loss': self.ctc_loss}
        )


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = VoiceToTextModel(num_features=80, max_text_length=200)
    
    # Example training data (you would need actual audio files and transcriptions)
    audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
    transcriptions = ['hello world', 'machine learning', 'speech recognition']
    
    # Train the model
    # history = model.train(audio_files, transcriptions, epochs=50)
    
    # Make predictions
    # predicted_text = model.predict('test_audio.wav')
    # print(f"Predicted text: {predicted_text}")
    
    print("Voice-to-Text model initialized successfully!")
    print("Model architecture:")
    if model.model:
        model.model.summary()
