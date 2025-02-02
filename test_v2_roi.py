import keras_ocr
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re

# Initialize the Keras OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Load the image
image_path = 'data/screenshots/screenshot.png'  # Change this to your image path
image = keras_ocr.tools.read(image_path)

# Image Preprocessing
# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply thresholding to create a binary image
_, thresholded_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY_INV)

# Create a copy of the image to draw on
image_with_boxes = image.copy()

# Define key phrases to find (including special characters)
key_phrases = ["Groceries & Pets", "Watches", "Automotive"]

# Create a pattern to match any of the key phrases, specifically looking for ampersands
pattern = r'|'.join([re.escape(phrase.lower()).replace(r'\&', r'[\s]*&[\s]*') for phrase in key_phrases])
print("Compiled regex pattern:", pattern)  # Debug: print the compiled pattern

# Use the pipeline to recognize text
prediction_groups = pipeline.recognize([image])

# Extract boxes and texts
boxes = []
recognized_texts = []  # To store recognized text for debugging
for predictions in prediction_groups:
    for text, box in predictions:
        recognized_texts.append(text)  # Store recognized text for debugging
        # Normalize recognized text
        normalized_text = re.sub(r'\s+', ' ', text.strip()).lower()  # Normalize and convert to lowercase
        # Check if the whole text matches the pattern
        if re.search(pattern, normalized_text):
            boxes.append((text, box))  # Append the original text and the box

# Debug: print out recognized texts
print("Recognized texts:", recognized_texts)

# Draw the bounding boxes for the matching phrases
for text, box in boxes:
    box = box.astype(int)  # Ensure coordinates are integers
    cv2.polylines(image_with_boxes, [box], isClosed=True, color=(0, 255, 0), thickness=2)

# If you want to find the nearest text to the matching phrases
if boxes:
    # Calculate the center of the boxes
    centers = np.array([np.mean(box, axis=0) for _, box in boxes])
    
    # Calculate distances to all recognized boxes
    distances = []
    for text, box in boxes:
        box_center = np.mean(box, axis=0)
        distances.append((text, box, np.min(np.linalg.norm(centers - box_center, axis=1))))
    
    # Find the nearest text to the matching phrases
    if distances:
        nearest_text = min(distances, key=lambda x: x[2])  # Get the nearest text
        nearest_box = nearest_text[1]
        nearest_box = nearest_box.astype(int)
        cv2.polylines(image_with_boxes, [nearest_box], isClosed=True, color=(255, 0, 0), thickness=2)  # Highlight nearest text

# Convert the image from BGR to RGB format for displaying
image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

# Display the image with recognized text
plt.imshow(image_with_boxes)
plt.axis('off')  # Hide axes

# Save the image with bounding boxes at a higher resolution
output_path = 'data/screenshots/result_with_boxes.png'  # Change this to your desired output path
plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Set dpi for higher resolution
plt.close()  # Close the plot to avoid displaying it again

print(f"Saved the image with highlighted text to {output_path} at higher resolution.")