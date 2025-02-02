import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import re

# Load the image
image_path = 'data/screenshots/screenshot.png'  # Change this to your image path
image = cv2.imread(image_path)

# Image Preprocessing
# Convert to grayscale (optional, depending on the input images)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise (optional, depending on the input images)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Define key phrases to find (including special characters)
key_phrases = ["Groceries & Pets", "Watches", "Automotive","Women's Bags","Fashion"]

# Create a pattern to match any of the key phrases, specifically looking for ampersands
pattern = r'|'.join([re.escape(phrase.lower()).replace(r'\&', r'[\s]*&[\s]*') for phrase in key_phrases])
print("Compiled regex pattern:", pattern)  # Debug: print the compiled pattern

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the language

# Use EasyOCR to recognize text on the image
results = reader.readtext(image)

# Extract boxes and texts
boxes = []
for (bbox, text, prob) in results:
    # Normalize recognized text
    normalized_text = re.sub(r'\s+', ' ', text.strip().lower())  # Normalize and convert to lowercase
    # Check if the whole text matches the pattern
    if re.search(pattern, normalized_text):
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x_min = int(min(top_left[0], bottom_left[0]))
        y_min = int(min(top_left[1], top_right[1]))
        w = int(max(top_right[0], bottom_right[0]) - x_min)
        h = int(max(bottom_left[1], bottom_right[1]) - y_min)
        boxes.append((text, (x_min, y_min, w, h)))  # Append the original text and the bounding box coordinates

# Draw the bounding boxes for the matching phrases
for text, (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle around the text

# Convert the image from BGR to RGB format for displaying
image_with_boxes = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with recognized text
plt.imshow(image_with_boxes)
plt.axis('off')  # Hide axes

# Save the image with bounding boxes at a higher resolution
output_path = 'data/screenshots/result_with_boxes_easyocr.png'  # Change this to your desired output path
plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Set dpi for higher resolution
plt.close()  # Close the plot to avoid displaying it again

print(f"Saved the image with highlighted text to {output_path} at higher resolution.")