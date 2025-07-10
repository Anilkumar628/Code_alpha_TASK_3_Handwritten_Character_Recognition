#Code_alpha_TASK_3_Handwritten_Character_Recognition
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Load EMNIST dataset
# Note: EMNIST is not included in Keras by default. Use EMNIST Letters (for a-z)
import tensorflow_datasets as tfds
emnist_data, info = tfds.load('emnist/letters', with_info=True, as_supervised=True)

# Split dataset
train_data, test_data = emnist_data['train'], emnist_data['test']

# Preprocess function
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [28, 28, 1])  # Add channel dimension
    label = label - 1  # Convert labels from 1-26 to 0-25 (for 'a' to 'z')
    return image, label

train_data = train_data.map(preprocess).batch(128).shuffle(10000)
test_data = test_data.map(preprocess).batch(128)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')  # 26 classes for a-z
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=5, validation_data=test_data)

# Evaluate
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f}")

# Sample Prediction
for image, label in test_data.take(1):
    predictions = model.predict(image)
    for i in range(5):
        plt.imshow(tf.squeeze(image[i]), cmap='gray')
        plt.title(f"True: {chr(label[i].numpy()+65)}, Predicted: {chr(np.argmax(predictions[i])+65)}")
        plt.axis('off')
        plt.show()
