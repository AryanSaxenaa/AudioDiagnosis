import tensorflow as tf
import pathlib

# --- Configuration ---
KERAS_MODEL_PATH = 'best_washing_machine_model.h5'
TFLITE_MODEL_DIR = 'tflite_models'
TFLITE_MODEL_NAME = 'audio_diagnosis_model'

# --- 1. Load the Trained Keras Model ---
# This loads the architecture, weights, and optimizer state.
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
print("Keras model loaded successfully.")

# --- 2. Create the Converter ---
# The converter is initialized from our loaded Keras model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- 3. (Optional but Recommended) Apply Optimizations ---
# This is where we apply quantization.
# The DEFAULT optimization includes float16 quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("Applying default optimizations (including Float16 quantization).")

# --- 4. Convert the Model ---
# This performs the actual conversion.
tflite_model = converter.convert()
print("Model converted to TensorFlow Lite format.")

# --- 5. Save the TensorFlow Lite Model ---
# Create the directory if it doesn't exist
tflite_model_dir = pathlib.Path(TFLITE_MODEL_DIR)
tflite_model_dir.mkdir(exist_ok=True)

# Define the full path for the .tflite file
tflite_model_path = tflite_model_dir / f"{TFLITE_MODEL_NAME}.tflite"

# Write the model to a file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("-" * 50)
print(f"✅ TensorFlow Lite model saved successfully to: {tflite_model_path}")
print(f"File size: {tflite_model_path.stat().st_size / 1024:.2f} KB")
print("-" * 50)
print("Next step: Add this .tflite file to your Flutter project's assets folder.")