from keras.applications.resnet import ResNet50
from keras.applications.resnet import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

# Path to the image to predict
img_path = '/home/babayaga/Documents/OCR/Computer Vision/images/test3.jpg'

# Load the image
img = load_img(img_path)

# Resize the image to a 224x224 square shape
img = img.resize((224, 224))

# Convert the image to an array
img_array = img_to_array(img)

# Convert the image into a 4-dimensional Tensor
# Convert from (height, width, channels) to (batch size, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the input image array using ImageNet preprocessing
img_array = imagenet_utils.preprocess_input(img_array)

# Load the pre-trained ResNet50 model with ImageNet weights
pretrained_model = ResNet50(weights="imagenet")

# Make a prediction using the pre-trained model
prediction = pretrained_model.predict(img_array)

# Decode the prediction into human-readable labels
actual_prediction = imagenet_utils.decode_predictions(prediction)

# Print the predicted object and its accuracy
print("Predicted object is:")
print(actual_prediction[0][0][1])
print("with accuracy")
print(actual_prediction[0][0][2] * 100)

# Display the image and the prediction text over it
disp_img = cv2.imread(img_path)

# Display the prediction text over the image
cv2.putText(disp_img, actual_prediction[0][0][1], (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))

# Show the image with prediction
cv2.imshow("Prediction", disp_img)
