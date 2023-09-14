from PIL import Image
import pytesseract
import cv2
import re

# Load the image from the disk
image_to_ocr = cv2.imread('/home/babayaga/Documents/OCR/Computer Vision/images/testing/fox_sample2.png')

# Preprocess the image
# Step 1: Convert to grayscale
preprocessed_img = cv2.cvtColor(image_to_ocr, cv2.COLOR_BGR2GRAY)

# Step 2: Apply binary and Otsu thresholding
preprocessed_img = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Step 3: Smooth the image using median blur
preprocessed_img = cv2.medianBlur(preprocessed_img, 3)

# Save the preprocessed image temporarily to disk
cv2.imwrite('temp_img.jpg', preprocessed_img)

# Read the temporary image from disk as a PIL image
preprocessed_pil_img = Image.open('temp_img.jpg')

# Pass the PIL image to Tesseract to perform OCR
text_extracted = pytesseract.image_to_string(preprocessed_pil_img)

# Remove non-printable ASCII characters
text_extracted = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text_extracted)

print("The extracted text from the image is:\n")
print(text_extracted)
