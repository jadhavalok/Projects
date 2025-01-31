import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16

# Define input shape
input_shape = (256, 256, 3)

# Define the number of classes
num_classes = 3

# Load the pre-trained VGG16 model without the top (classification) layers
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the pre-trained layers to prevent them from being updated during training
for layer in vgg16_base.layers:
    layer.trainable = False

# Add classification layers on top of the VGG16 base
x = Flatten()(vgg16_base.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

# Create the VGG16 model
vgg16_model = Model(inputs=vgg16_base.input, outputs=outputs)

# Compile the model with Nadam optimizer
vgg16_model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
vgg16_model.summary()

# Define data generators with data augmentation for regularization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\Dataset2.0\\train',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\Dataset2.0\\validation',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical'
)

# Test data generator
test_generator = test_datagen.flow_from_directory(
    'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\Dataset2.0\\test',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical'
)

# Train the VGG16 model with early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_vgg16 = vgg16_model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping])

# Save the VGG16 model
vgg16_model.save('vgg16_model_a.h5')

# Evaluate the VGG16 model on the test data
test_loss_vgg16, test_accuracy_vgg16 = vgg16_model.evaluate(test_generator)
print("VGG16 Test Loss:", test_loss_vgg16)
print("VGG16 Test Accuracy:", test_accuracy_vgg16)

# Get predictions for the test data using VGG16 model
test_predictions_vgg16 = vgg16_model.predict(test_generator)
test_predictions_vgg16 = np.argmax(test_predictions_vgg16, axis=1)

# Get true labels for the test data
test_true_labels = test_generator.classes

# Calculate confusion matrix for VGG16 model
conf_matrix_vgg16 = confusion_matrix(test_true_labels, test_predictions_vgg16)

# Print confusion matrix for VGG16 model
print("VGG16 Confusion Matrix:")
print(conf_matrix_vgg16)

# Calculate and print classification report for VGG16 model
class_names = list(test_generator.class_indices.keys())
print("VGG16 Classification Report:")
print(classification_report(test_true_labels, test_predictions_vgg16, target_names=class_names))
