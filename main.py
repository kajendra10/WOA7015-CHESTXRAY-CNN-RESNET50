import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
from keras._tf_keras.keras import models, layers
from keras.src.applications.resnet import ResNet50
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from keras._tf_keras import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import Model
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Path to dataset
normal_path = r"C:\Users\USER\PycharmProjects\WOA7015GroupAssignment\chest_xray\chest_xray\train\NORMAL"
pneumonia_path = r"C:\Users\USER\PycharmProjects\WOA7015GroupAssignment\chest_xray\chest_xray\train\PNEUMONIA"

images = []
labels = []

# Function to load and preprocess images
def load_images_from_directory(directory, label):
    for img_file1 in os.listdir(directory):
        img_path1 = os.path.join(directory, img_file1)
        try:
            img1 = Image.open(img_path1).resize((224, 224))  # Resize to match CNN input size
            if img1.mode != 'RGB':  # Convert grayscale to RGB
                img1 = img1.convert('RGB')
            img_array1 = np.array(img1)
            if img_array1.shape == (224, 224, 3):  # Ensure the image has correct shape
                images.append(img_array1 / 255.0)  # Normalize pixel values
                labels.append(label)  # Append correct label
            else:
                print(f"Skipping invalid image {img_path1} with shape {img_array1.shape}")
        except Exception as E:
            print(f"Error processing {img_path1}: {E}")

# Load images
load_images_from_directory(normal_path, 0)  # Label 0 for NORMAL
load_images_from_directory(pneumonia_path, 1)  # Label 1 for PNEUMONIA

# Convert lists into numpy arrays
images = np.array(images, dtype=np.float32)
labels = np.array(labels)

# Check data size and shape
print(f"Loaded {len(images)} images with shape {images.shape} and {len(np.unique(labels))} classes.")

# Split data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# Load ResNet50 pre-trained on ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the pre-trained layers

base_model.trainable = True
for layer in base_model.layers[:140]:  # Freeze the initial layers
    layer.trainable = False

# Build CNN model
model = models.Sequential([
    base_model,  # Pre-trained ResNet50 as base
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Reduced learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
base_model = model.get_layer('resnet50')
for layer in base_model.layers:
    print(layer.name)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss', # Metric to monitor for early stopping
    patience=5, # Wait for 5 epochs without improvement
    restore_best_weights=True # Restore the best weights after stopping
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,  # Adjust epochs based on performance
    batch_size=32,  # Batch size
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1  # Display training progress
)

# Model Evaluation
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.4f}")

# Graph Plot for Train & Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Graph Plot for Train & Validation Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Paths to test dataset
test_normal_path = r"C:\Users\USER\PycharmProjects\WOA7015GroupAssignment\chest_xray\chest_xray\test\NORMAL"
test_pneumonia_path = r"C:\Users\USER\PycharmProjects\WOA7015GroupAssignment\chest_xray\chest_xray\test\PNEUMONIA"

# Initialize lists for test images and labels
test_images = []
test_labels = []

# Function to load and preprocess test images
def load_test_images_from_directory(directory, label):
    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        try:
            img = Image.open(img_path).resize((224, 224))  # Resize to match CNN input size
            if img.mode != 'RGB':  # Convert grayscale to RGB
                img = img.convert('RGB')
            img_array = np.array(img)
            if img_array.shape == (224, 224, 3):  # Ensure the image has correct shape
                test_images.append(img_array / 255.0)  # Normalize pixel values
                test_labels.append(label)  # Append correct label
            else:
                print(f"Skipping invalid image {img_path} with shape {img_array.shape}")
        except Exception as E:
            print(f"Error processing {img_path}: {E}")

# Load test images
load_test_images_from_directory(test_normal_path, 0)  # Label 0 for NORMAL
load_test_images_from_directory(test_pneumonia_path, 1)  # Label 1 for PNEUMONIA

# Convert lists into numpy arrays
test_images = np.array(test_images, dtype=np.float32)
test_labels = np.array(test_labels)

# New code
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# New code
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)


# Check data size and shape
print(f"Loaded {len(test_images)} test images with {len(np.unique(test_labels))} classes.")

# Model Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.4f}")

# Generate Predictions
predictions = model.predict(test_images)
predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels


# Visualising Results by Generating Confusion Matrix
cm = confusion_matrix(test_labels, predicted_classes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
disp.plot(cmap="Blues")
plt.show()

# Saliency Map Function
def generate_saliency_map(model, img_array):
    # Ensure the input image has a batch dimension
    img_array = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        tape.watch(img_array)  # Watch the input image
        predictions = model(img_array)
        loss = predictions[:, 0]  # Focus on the positive class

    # Calculate gradients of the loss w.r.t. the input image
    grads = tape.gradient(loss, img_array)

    # Compute the absolute value of the gradients and take the maximum along the color channels
    saliency = np.max(np.abs(grads), axis=-1)[0]

    # Normalize the saliency map to range [0, 1]
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) + 1e-8)
    return saliency


# Loop over test images
for i, img in enumerate(test_images[:5]):  # Adjust the number of images to display
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension

    # Generate Saliency Map
    saliency_map = generate_saliency_map(model, img_array)

    # Visualize Saliency Map
    plt.figure(figsize=(15, 7))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow((img * 255).astype(np.uint8))
    plt.axis('on')
    plt.title(f'Original X-Ray {i + 1}')

    # Saliency Map
    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map, cmap='jet')
    plt.axis('on')
    plt.title(f'Saliency Map Classification {i + 1}')

    plt.show()


    # Occlusion Sensitivity Function
    def generate_occlusion_map(model, img_array, patch_size=20, stride=10):
        # Create a copy of the input image
        img_copy = np.copy(img_array)

        # Get image dimensions
        img_height, img_width, _ = img_array.shape

        # Initialize an empty array to store the occlusion results
        occlusion_map = np.zeros((img_height, img_width))

        # Get the model prediction for the original image (baseline prediction)
        baseline_prediction = model.predict(np.expand_dims(img_array, axis=0))

        # Loop over the image and occlude patches
        for i in range(0, img_height - patch_size, stride):
            for j in range(0, img_width - patch_size, stride):
                # Occlude the current patch
                img_occluded = np.copy(img_copy)
                img_occluded[i:i + patch_size, j:j + patch_size, :] = 0  # Set patch to black (0)

                # Predict with the occluded image
                occluded_prediction = model.predict(np.expand_dims(img_occluded, axis=0))

                # Calculate the difference between the baseline and occluded prediction
                # We use the probability of the correct class (assuming binary classification)
                baseline_class_prob = baseline_prediction[0][0]
                occluded_class_prob = occluded_prediction[0][0]
                occlusion_map[i:i + patch_size, j:j + patch_size] = abs(baseline_class_prob - occluded_class_prob)

            # Normalize the occlusion map
            occlusion_map = occlusion_map / np.max(occlusion_map)
            # occlusion_map = occlusion_map / np.max(occlusion_map) if np.max(occlusion_map) > 0 else occlusion_map
            return occlusion_map


    # Loop over test images to generate and visualize Occlusion Sensitivity Map
    for h, img in enumerate(test_images[:5]):  # Adjust the number of images to display
        img_array = np.expand_dims(img, axis=0)  # Add batch dimension

        # Generate Occlusion Sensitivity Map
        occlusion_map = generate_occlusion_map(model, img)

        # Visualize Occlusion Sensitivity Map
        plt.figure(figsize=(15, 7))

        # Original X-Ray Image
        plt.subplot(1, 2, 1)
        plt.imshow((img * 255).astype(np.uint8))
        plt.axis('on')
        plt.title(f'Original X-Ray {h + 1}')

        # Occlusion Sensitivity Map
        plt.subplot(1, 2, 2)
        plt.imshow(occlusion_map, cmap='jet')
        plt.axis('on')
        plt.title(f'Occlusion Sensitivity Map {h + 1}')
        plt.show()

















