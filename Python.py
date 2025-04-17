import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

DATASET_PATH = "*******************"
IMAGE_DIR = os.path.join(DATASET_PATH, "train") 
LABEL_FILE = os.path.join(IMAGE_DIR, "_classes.csv") 

def load_labels(label_file):
    df = pd.read_csv(label_file, header=None)
    df.columns = ['filename', 'brain-hemorrhage', 'brain-infarct', 'brain-normal']
    return df

def prepare_data(df, image_dir, image_size=(640, 640)):
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size) / 255.0
            images.append(image)

          
            label = row[['barin-hemorrhage', 'barin-infarct', 'barin-normal']].values.astype(int)
            labels.append(label)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)


df = load_labels(LABEL_FILE)
X, y = prepare_data(df, IMAGE_DIR)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер тренировочного набора: {X_train.shape}")
print(f"Размер тестового набора: {X_val.shape}")

def visualize_sample(image, label):
    plt.imshow(image)
    class_names = ['barin-hemorrhage', 'barin-infarct', 'barin-normal']
    label_text = ", ".join([name for idx, name in enumerate(class_names) if label[idx] == 1])
    plt.title(f"Class: {label_text}")
    plt.axis('off')
    plt.show()


visualize_sample(X_train[0], y_train[0])


model = models.Sequential([
    layers.InputLayer(input_shape=(640, 640, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 класса 
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Тестовая точность: {accuracy * 100:.2f}%")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("Точность модели на обучении и валидации")
plt.show()
