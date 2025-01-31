import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Define function to load and preprocess image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Define function to classify terrain type using your trained model
def classify_terrain(image_path, model):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    terrain_classes = ['Mountains', 'Desert', 'Forest Cover']  # Update with your classes
    terrain_type = terrain_classes[np.argmax(predictions)]
    return terrain_type

# Define function to retrieve static information about terrain types
def get_terrain_information(terrain_type):
    # Define information for each terrain type
    # You can add more details based on your requirements
    terrain_information = {
        'Mountains': {
            'Soil Type': 'Rocky, Sandy, Loamy',
            'Average Rainfall': 'Varies depending on altitude and location',
            'Temperature Range': 'Cool to Cold',
            'Vegetation': 'Alpine vegetation, including coniferous forests, grasslands, and shrubs',
            'Wildlife': 'Diverse wildlife including bears, mountain goats, deer, and various bird species',
            'Human Population Density': 'Low',
            'Tourism and Recreation': 'Popular for hiking, climbing, skiing, and nature tourism',
        },
        'Desert': {
            'Soil Type': 'Sandy, Rocky, Clayey',
            'Average Rainfall': 'Low',
            'Temperature Range': 'Hot during day, cold at night',
            'Vegetation': 'Sparse vegetation, including cacti, succulents, and drought-resistant plants',
            'Wildlife': 'Adapted to arid conditions, including reptiles, insects, and small mammals',
            'Human Population Density': 'Varies',
            'Economic Activities': 'Mining, tourism, and agriculture',
        },
        'Forest Cover': {
            'Soil Type': 'Loamy, Sandy, Clayey',
            'Average Rainfall': 'Moderate to High',
            'Temperature Range': 'Mild to Cool',
            'Vegetation': 'Dense vegetation including trees, shrubs, and various plant species',
            'Wildlife': 'Rich biodiversity including mammals, birds, insects, and amphibians',
            'Human Population Density': 'Varies',
            'Economic Importance': 'Forests provide timber, fuelwood, biodiversity, and ecosystem services',
        }
    }
    return terrain_information.get(terrain_type, {})

# Load your trained model
# Replace 'your_model.h5' with the path to your trained model file
model = tf.keras.models.load_model('C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Terrain\\resnet_mode1l.h5')

# Create Tkinter window
window = tk.Tk()
window.title("Terrain Classification")

# Define function to handle file selection
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((256, 256))  # Remove 'ANTIALIAS' attribute
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        predicted_terrain = classify_terrain(file_path, model)
        terrain_information = get_terrain_information(predicted_terrain)
        terrain_info_text.config(state=tk.NORMAL)
        terrain_info_text.delete('1.0', tk.END)
        terrain_info_text.insert(tk.END, f"Terrain Type: {predicted_terrain}\n\nTerrain Information:\n")
        if terrain_information:
            for key, value in terrain_information.items():
                terrain_info_text.insert(tk.END, f"{key}: {value}\n")
        else:
            terrain_info_text.insert(tk.END, "Terrain type could not be determined.")
        terrain_info_text.config(state=tk.DISABLED)

# Create GUI components
select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack(pady=10)

image_label = tk.Label(window)
image_label.pack()

terrain_info_text = tk.Text(window, wrap=tk.WORD, height=10, width=50)
terrain_info_text.pack()

window.mainloop()
