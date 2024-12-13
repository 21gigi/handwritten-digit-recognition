# Handwritten Digit Recognition

This project is a Python-based application that uses a Convolutional Neural Network (CNN) to recognize handwritten digits. The application provides both an interactive graphical user interface (GUI) for image-based digit recognition and a real-time camera-based digit recognition feature.

---

## **Getting Started**

### **Prerequisites**
Ensure you have Python 3.8 or later installed on your system. Install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

---

## **Running the Application**

1. Save the project code to a file named `digit_recognition.py`.
2. Open your terminal and navigate to the folder containing the script.
3. Run the application using the following command:

```bash
python digit_recognition.py
```

---

## **User Interface Instructions**

### **Loading an Image**
1. Click the **"Load Image"** button in the GUI.
2. Select an image file (`.png`, `.jpg`, or `.jpeg`) from your file explorer.
3. The selected image will appear in the application window.
4. Use your mouse to draw a rectangle around the region of interest (ROI).
5. Click the **"Predict"** button to analyze the selected region. The application will display the predicted digit and the corresponding confidence scores.

### **Using the Camera**
1. Click the **"Start Camera"** button to activate the webcam.
2. A live video feed will appear with a green rectangle marking the region of interest.
3. Position a handwritten digit within the rectangle for analysis.
4. The predicted digit will appear as green text on the video feed.
5. Press the `q` key to stop the camera feed.

---

## **Key Features**

1. **Interactive GUI**: Provides an easy-to-use interface for loading images, cropping regions of interest, and predicting digits.
2. **Real-Time Camera Recognition**: Enables live recognition of handwritten digits using a webcam.
3. **Accurate Predictions**: Powered by a Convolutional Neural Network trained on the MNIST dataset, achieving high accuracy.

---

## **Project Overview**

### **Model Architecture**
- **Input Shape**: 28x28 grayscale images
- **Layers**:
  - Convolutional (32 filters, 3x3 kernel, ReLU activation)
  - MaxPooling (2x2)
  - Convolutional (64 filters, 3x3 kernel, ReLU activation)
  - MaxPooling (2x2)
  - Flatten
  - Dense (64 units, ReLU activation)
  - Dense (10 units, Softmax activation for classification)

### **Training**
The model is trained on the MNIST dataset for 5 epochs using the Adam optimizer and sparse categorical cross-entropy loss.

---

## **Troubleshooting**

1. **ModuleNotFoundError**:
   - Ensure all required libraries are installed using `pip install -r requirements.txt`.
2. **Camera Feed Issues**:
   - Check that your webcam is properly connected and accessible.
3. **Low Accuracy**:
   - Ensure the digit is clear and centered within the region of interest.

---

## **Acknowledgments**
This project utilizes the MNIST dataset, a widely used benchmark dataset for image classification and machine learning.

The file `photo_2.jpg` used in this project is sourced from the [Handwritten Digit Recognition GitHub repository](https://github.com/ehsanmqn/handwritten-digit-recognition/blob/master/photo_2.jpg).

This project is the final project of third-year Computer Science students:
- Nicole Bermundo
- Angela Cabanes
- Zynnon Kyle Depoo
- Michaela Jornales

under the Artificial Intelligence course.

---

## **License**
This project is open-source and available under the MIT License.
