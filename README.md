# Image-Processing-and-CNN-with-OpenCV-and-TensorFlow
This project demonstrates various image processing techniques using **OpenCV** and builds a Convolutional Neural Network (CNN) for image classification using the **Fashion MNIST** and **MNIST** datasets. It includes image handling, transformations, and CNN model creation for multi-class classification.
## Project Overview
1. **Image Processing**: Various techniques such as image reading, cropping, resizing, blurring, and manipulation using OpenCV.
2. **CNN for Image Classification**: A Convolutional Neural Network built using TensorFlow/Keras to classify images from the **Fashion MNIST** and **MNIST** datasets.
## Technologies Used
- **Python**: Programming language.
- **OpenCV**: Library for image processing.
- **NumPy**: Used for array manipulation.
- **Matplotlib**: Plotting and visualization.
- **TensorFlow/Keras**: Framework for building the CNN model.
- **scikit-learn**: Metrics for model evaluation (confusion matrix, classification report).
## Key Features
1. **Image Processing**:
   - Loading and displaying images in colored and grayscale formats.
   - Cropping, resizing, and blurring images.
   - Creating and manipulating images using NumPy arrays.
   - Inpainting: Adding shapes and text to images.
2. **CNN for Fashion MNIST and MNIST Datasets**:
   - Builds a CNN with three convolutional layers, followed by max-pooling and fully connected layers.
   - Model training, validation, and evaluation using accuracy and loss metrics.
   - Visualizes model performance with loss and accuracy curves.
   - Generates predictions and evaluates model using a confusion matrix.
## CNN Model Architecture
- **Conv2D Layer 1**: 64 filters, 3x3 kernel, ReLU activation.
- **MaxPooling2D Layer 1**: 2x2 pool size.
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation.
- **MaxPooling2D Layer 2**: 2x2 pool size.
- **Conv2D Layer 3**: 32 filters, 3x3 kernel, ReLU activation.
- **MaxPooling2D Layer 3**: 2x2 pool size.
- **Flatten**: Converts 2D feature maps to 1D.
- **Dense Layer**: Fully connected layer with 32 units, ReLU activation.
- **Output Layer**: Fully connected layer with 10 units, Softmax activation (for 10-class classification).
## Results
The model is trained for 25 epochs, and the results are plotted for both loss and accuracy over time. The performance is evaluated on the test dataset with a confusion matrix and classification report.
### Sample Accuracy & Loss Plots:
- **Training Loss vs. Validation Loss**
- **Training Accuracy vs. Validation Accuracy**
### Confusion Matrix Example:
The confusion matrix shows the true vs. predicted labels for the test set.
## Future Improvements
- Fine-tuning the CNN model architecture to improve performance.
- Adding more advanced image augmentation techniques to the training data.
- Experimenting with deeper networks or pre-trained models for better accuracy.
- Implementing other datasets for more generalized models.
