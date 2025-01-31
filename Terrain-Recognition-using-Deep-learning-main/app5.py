import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from joblib import load

class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier")
        
        self.label = tk.Label(self.master, text="Select an image:")
        self.label.pack()
        
        self.load_button = tk.Button(self.master, text="Load Image", command=self.load_image)
        self.load_button.pack()
        
        self.predict_button = tk.Button(self.master, text="Predict", command=self.predict)
        self.predict_button.pack()
        
        self.image_label = tk.Label(self.master)
        self.image_label.pack()
        
        self.model = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
        if file_path:
            self.image = Image.open(file_path)
            self.image.thumbnail((300, 300))  # Resize image to fit in the UI
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)

    def predict(self):
        if self.image:
            if not self.model:
                self.model = load("random_forest_model.joblib")  # Load the model
            # Preprocess the image
            resized_image = self.image.resize((256, 256))  # Resize image to match model input size
            image_array = np.array(resized_image) / 255.0  # Normalize pixel values
            flattened_image = image_array.flatten()  # Flatten image
            # Make prediction
            prediction = self.model.predict([flattened_image])  # Wrap flattened image in a list
            # Display prediction
            class_names = ['Deserts', 'Forest Cover', 'Mountains']  # Assuming these are the class labels
            predicted_class = class_names[int(prediction)]
            messagebox.showinfo("Prediction", f"The predicted class is: {predicted_class}")

def main():
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
