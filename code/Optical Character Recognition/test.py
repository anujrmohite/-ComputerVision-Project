import pytesseract
import pkg_resources
import cv2

#printing the tesseract and opencv version
print("pytesseract version:",pkg_resources.working_set.by_key['pytesseract'].version)
print("cv2 version:",cv2.__version__)

