import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dropout, BatchNormalization, Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# Load the data.
df = pd.read_csv('hubert_self_attentive_features.csv')

# Set train/test patient_number.
test_ids = [1981, 68, 11, 1159, 2097]
train_ids = [
    1205, 1449, 15, 29, 32, 4, 40, 41, 43, 5, 53, 59, 60, 61, 63, 66, 67,
    69, 74, 9, 1197, 1203, 1694, 1819, 2121, 2377, 2509, 2564, 2596, 2601,
    816, 846
]

df_train = df[df['patient_number'].isin(train_ids)].reset_index(drop=True)
df_test  = df[df['patient_number'].isin(test_ids)].reset_index(drop=True)

# Prepare feature and label columns.
exclude_cols = {'disease_label', 'file_name', 'sound_type', 'patient_number'}
feature_cols = [c for c in df.columns if c not in exclude_cols]

X_train = df_train[feature_cols].values.astype(float)
y_train = df_train['disease_label'].values

X_test  = df_test[feature_cols].values.astype(float)
y_test  = df_test['disease_label'].values

# Encoding the labels.
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

# Standardize.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# Reshape to LSTM input format (samples, timesteps, features=1).
timesteps = X_train.shape[1]
X_train = X_train.reshape(-1, timesteps, 1)
X_test  = X_test.reshape(-1, timesteps, 1)

# Calculating Class Weights.
cw = class_weight.compute_class_weight('balanced',
                                       classes=np.unique(y_train),
                                       y=y_train)
class_weights = dict(enumerate(cw))

# Building a Bidirectional LSTM Model.
tf.random.set_seed(42)
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(timesteps, 1)),
    Dropout(0.3),
    BatchNormalization(),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Set callbacks.
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ModelCheckpoint('best_hubert_lstm.keras', monitor='val_accuracy', save_best_only=True)
]

# Training (20% validation set).
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# Evaluate on the specified test set.
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss for New HuBERT Features: {loss:.4f}, Test Accuracy for New HuBERT Features: {acc:.4f}')

