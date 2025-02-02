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

# Use the pipeline to recognize text
prediction_groups = pipeline.recognize([image])

# Create a copy of the image to draw on
image_with_boxes = image.copy()

# Define the key phrases you want to find (normalized)
key_phrases = ["Automotive", "Watches"]
normalized_key_phrases = [phrase.lower().strip() for phrase in key_phrases]

# Extract boxes and texts
boxes = []
for predictions in prediction_groups:
    for text, box in predictions:
        boxes.append((text, box))

# Normalize recognized text and filter for key phrases
target_boxes = []
for text, box in boxes:
    normalized_text = re.sub(r'\s+', ' ', text.lower().strip())  # Normalize recognized text
    if any(normalized_text in phrase for phrase in normalized_key_phrases):
        target_boxes.append((text, box))

# Draw the bounding boxes for the target phrases
for text, box in target_boxes:
    box = box.astype(int)  # Ensure coordinates are integers
    cv2.polylines(image_with_boxes, [box], isClosed=True, color=(0, 255, 0), thickness=2)

# If you want to find the nearest text to the target phrases
if target_boxes:
    # Calculate the center of the target boxes
    target_centers = np.array([np.mean(box, axis=0) for _, box in target_boxes])
    
    # Calculate distances to all recognized boxes
    distances = []
    for text, box in boxes:
        # Normalize the recognized text
        normalized_text = re.sub(r'\s+', ' ', text.lower().strip())  # Normalize recognized text
        if not any(normalized_text in target[0].lower() for target in target_boxes):  # Skip already selected boxes
            box_center = np.mean(box, axis=0)
            # Calculate distance to each target center
            distances.append((text, box, np.min(np.linalg.norm(target_centers - box_center, axis=1))))
    
    # Find the nearest text to the target phrases
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