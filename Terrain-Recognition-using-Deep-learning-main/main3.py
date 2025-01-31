import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Define input shape
input_shape = (256, 256, 3)

# Define the number of classes
num_classes = 3

def resnet_block(input_tensor, filters, kernel_size, strides=(1, 1), activation='relu'):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Check if input_tensor needs to be resized
    if input_tensor.shape[-1] != filters or strides != (1, 1):
        input_tensor = Conv2D(filters, (1, 1), strides=strides, padding='same')(input_tensor)
    
    x = Add()([x, input_tensor])
    x = Activation(activation)(x)
    return x

# Define the RESNET model
def resnet_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)  # Shape: (128, 128, 64)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # Shape: (64, 64, 64)

    x = resnet_block(x, 64, (3, 3))  # Expecting input shape: (64, 64, 64)
    x = resnet_block(x, 64, (3, 3))  # Expecting input shape: (64, 64, 64)
    x = resnet_block(x, 64, (3, 3))  # Expecting input shape: (64, 64, 64)

    x = resnet_block(x, 128, (3, 3), strides=(2, 2))  # Expecting input shape: (32, 32, 128)
    x = resnet_block(x, 128, (3, 3))  # Expecting input shape: (32, 32, 128)
    x = resnet_block(x, 128, (3, 3))  # Expecting input shape: (32, 32, 128)

    x = resnet_block(x, 256, (3, 3), strides=(2, 2))  # Expecting input shape: (16, 16, 256)
    x = resnet_block(x, 256, (3, 3))  # Expecting input shape: (16, 16, 256)
    x = resnet_block(x, 256, (3, 3))  # Expecting input shape: (16, 16, 256)

    x = AveragePooling2D(pool_size=(7, 7))(x)  # Shape: (1, 1, 256)
    x = Flatten()(x)  # Shape: (256,)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create the RESNET model
model = resnet_model(input_shape, num_classes)

# Compile the model with modified hyperparameters
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\Dataset 1.0\\train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\Dataset 1.0\\validation',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\Dataset 1.0\\test',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Implement early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(train_generator, 
                    epochs=20, 
                    validation_data=val_generator, 
                    callbacks=[early_stopping])

# Save the model
model.save('resnet_model.h5')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
