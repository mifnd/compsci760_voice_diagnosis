#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dropout, BatchNormalization, Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# Locate the directory of the current script.
script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
csv_path = os.path.join(repo_root, "data", "processed", "10mfcc_mean.csv")

# Read the features.
df = pd.read_csv(csv_path)

train_path = os.path.join(repo_root, "data", "train1.txt")
test_path  = os.path.join(repo_root, "data", "test1.txt")

# Read the patient_number of each row and convert it into an integer list.
with open(train_path, "r", encoding="utf-8") as f:
    train_ids = [
        int(line.strip())
        for line in f
        if line.strip() != ""
    ]

with open(test_path, "r", encoding="utf-8") as f:
    test_ids = [
        int(line.strip())
        for line in f
        if line.strip() != ""
    ]

df_train = df[df['patient_number'].isin(train_ids)].reset_index(drop=True)
df_test  = df[df['patient_number'].isin(test_ids)].reset_index(drop=True)

# Features and labels.
mfcc_feats = [f'mfcc_{i}' for i in range(1, 11)]
X_train = df_train[mfcc_feats].values
y_train = df_train['disease_label'].values

X_test  = df_test[mfcc_feats].values
y_test  = df_test['disease_label'].values

# Encode the label.
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

# Divide the validation set from the training set.
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# Standardize the training set.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Adjust to LSTM input format (samples, timesteps=10, features=1).
X_train = X_train.reshape(-1, 10, 1)
X_val   = X_val.reshape(-1,   10, 1)
X_test  = X_test.reshape(-1,   10, 1)

# Calculate class weights to handle imbalance.
cw = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(cw))

# Build a bidirectional two-layer LSTM + BN + Dropout model.
tf.random.set_seed(42)
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(10,1)),
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

# Set the callback functions.
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ModelCheckpoint('best_mfcc_lstm.keras',
                    monitor='val_accuracy',
                    save_best_only=True)
]

# Fit the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# Evaluate the model.
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'10mfcc_mean Test set → Loss: {loss:.4f}, Acc: {acc:.4f}')

