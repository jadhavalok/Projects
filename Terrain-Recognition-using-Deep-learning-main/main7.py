import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import InceptionV3

# Define input shape
input_shape = (256, 256, 3)

# Define the number of classes
num_classes = 3

# Create InceptionV3 model
base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)

# Add additional layers for classification
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=outputs)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Define data generators
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
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\Dataset2.0\\validation',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\Dataset2.0\\test',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Get predictions for the test data
test_predictions = model.predict(test_generator)
test_predictions = np.argmax(test_predictions, axis=1)

# Get true labels for the test data
test_true_labels = test_generator.classes

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_true_labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_names = list(test_generator.class_indices.keys())
print("Classification Report:")
print(classification_report(test_true_labels, test_predictions, target_names=class_names))

model.save("inceptionv3_a.h5")