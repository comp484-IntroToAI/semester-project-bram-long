from PIL import Image
import pytesseract

# Specify the path to the Tesseract executable if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'alphabetadelta.png'
image = Image.open(image_path)

# Use Tesseract to do OCR on the image
text = pytesseract.image_to_string(image)

# Print the extracted text
print("Extracted Text:")
print(text)
