import tensorflow as tf
from tensorflow.keras import layers, models
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# Load the pre-trained MNIST model
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)


def predict_image(image):
    image = image.resize((28, 28)).convert("L")  # Convert to grayscale and resize
    img_array = np.array(image) / 255.0         # Normalize
    img_array = img_array.reshape(1, 28, 28, 1) # Reshape for the model
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    return predicted_label, prediction[0]


def predict_camera_frame(frame):
    frame = cv2.resize(frame, (28, 28))         # Resize to model input size
    frame = frame / 255.0                       # Normalize
    frame = frame.reshape(1, 28, 28, 1)         # Reshape for the model
    prediction = model.predict(frame)
    predicted_label = np.argmax(prediction)
    return predicted_label


# GUI Application
class SnipApp:
    def __init__(self, root):
        self.root = root
        self.canvas = None
        self.rect = None
        self.start_x = self.start_y = 0
        self.crop_coords = None
        self.image = None
        self.camera_running = False

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.camera_button = tk.Button(root, text="Start Camera", command=self.toggle_camera)
        self.camera_button.pack()

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack()
        self.predict_button.config(state="disabled")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        image = Image.open(file_path)
        self.image = image
        self.tk_image = ImageTk.PhotoImage(image)

        if self.canvas:
            self.canvas.destroy()
        self.canvas = tk.Canvas(self.root, width=image.width, height=image.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        self.crop_coords = (int(x1), int(y1), int(x2), int(y2))
        self.predict_button.config(state="normal")

    def predict(self):
        if self.crop_coords and self.image:
            x1, y1, x2, y2 = self.crop_coords
            cropped_image = self.image.crop((x1, y1, x2, y2))
            predicted_label, probabilities = predict_image(cropped_image)
            messagebox.showinfo("Prediction", f"Predicted Label: {predicted_label}\nProbabilities: {probabilities}")
        else:
            messagebox.showwarning("Warning", "No image loaded or no crop region selected!")

    def toggle_camera(self):
        if self.camera_running:
            self.camera_running = False
            self.camera_button.config(text="Start Camera")
        else:
            self.camera_running = True
            self.camera_button.config(text="Stop Camera")
            self.run_camera()

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(frame, (220, 140), (420, 340), (0, 255, 0), 2)
            roi = gray_frame[140:340, 220:420]  # Region of interest for prediction
            predicted_label = predict_camera_frame(roi)

            cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Handwritten Digit Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.camera_running = False
        self.camera_button.config(text="Start Camera")


# Run the application
root = tk.Tk()
root.title("Handwritten Digit Recognition")
app = SnipApp(root)
root.mainloop()
