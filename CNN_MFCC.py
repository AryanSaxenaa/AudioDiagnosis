import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Step 1: Load the Preprocessed Data ---
DATA_PATH = 'extracted_audio_features.pkl'

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

features = data['features']
labels = data['labels']
label_encoder = data['label_encoder']

print(f"Data loaded successfully.")
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# --- Step 2: Prepare the Data for the CNN ---
features = np.expand_dims(features, -1)
print(f"Features shape after adding channel dimension: {features.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# --- Step 3: Build the CNN Model (Corrected & Refined) ---

# Get the input shape from the training data
input_shape = X_train.shape[1:]

# A simpler model with two convolutional blocks is more suitable for this input size.
model = Sequential([
    Input(shape=input_shape),

    # First Convolutional Block
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    # Batch Normalization helps stabilize and speed up training
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Second Convolutional Block
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # --- REMOVED THE THIRD CONV BLOCK THAT CAUSED THE ERROR ---

    # Flatten the features to feed into a dense layer
    Flatten(),

    # Dense Layer
    Dense(128, activation='relu'),
    Dropout(0.5),

    # Output Layer
    Dense(1, activation='sigmoid')
])

# Display the model's architecture
model.summary()


# --- Step 4: Compile the Model ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# --- Step 5: Train the Model ---
EPOCHS = 20
BATCH_SIZE = 32

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_washing_machine_model_20epochs.h5', save_best_only=True, monitor='val_accuracy')

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test), # It's better to validate on the test set
    callbacks=[early_stopping, model_checkpoint]
)


# --- Step 6: Evaluate the Model ---
print("\nEvaluating the best model on the test set...")
# Load the best model saved by ModelCheckpoint
model.load_weights('best_washing_machine_model.h5')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy*100:.2f}%')


# --- Step 7: Save the Final Trained Model ---
# No need to save again, as ModelCheckpoint already saved the best version.
print("\nBest model saved to 'best_washing_machine_model.h5'")