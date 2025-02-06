import cv2
import keras_ocr
import os
import numpy as np
from sklearn.cluster import KMeans
from transformers import pipeline

# Set environment variable to force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the image
image_path = 'data/screenshots/image.png'  # Change this to your image path
print("Loading image...")
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or could not be loaded.")
    exit()

# Initialize the Keras-OCR pipeline
ocr_pipeline = keras_ocr.pipeline.Pipeline()
print("Keras-OCR pipeline initialized.")

# Perform OCR using Keras-OCR
print("Performing OCR...")
images = [image]
prediction_groups = ocr_pipeline.recognize(images)
print("OCR completed.")

# Expected phrases to look for
expected_phrases = ["Groceries & Pets", "Watches", "Automotive", "Women's Bags", "Fashion"]

# Load the text correction model
print("Loading text correction model...")
corrector = pipeline("text2text-generation", model="facebook/bart-large-cnn")
print("Text correction model loaded.")

# Group boxes based on predictions
grouped_boxes = [(text, box) for text, box in prediction_groups[0] if text.strip()]
print(f"Grouped {len(grouped_boxes)} boxes from OCR predictions.")

# Apply text correction to the grouped boxes
corrected_boxes = []
for text, box in grouped_boxes:
    print(f"Correcting text: '{text}'")
    corrected_text = corrector(f"correct: {text}")[0]['generated_text']
    print(f"Corrected text: '{corrected_text}'")
    corrected_boxes.append((corrected_text, box))

# Determine number of clusters (can be adjusted)
n_clusters = max(1, len(corrected_boxes) // 2)
print(f"Determining clusters: {n_clusters} clusters.")

# Apply K-Means clustering
def cluster_boxes_kmeans(grouped_boxes, n_clusters):
    centers = np.array([get_box_center(box) for _, box in grouped_boxes])
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(centers)
    return kmeans.labels_

print("Clustering corrected boxes...")
labels = cluster_boxes_kmeans(corrected_boxes, n_clusters)
print("Clustering completed.")

# Draw boxes for all detected clusters
final_boxes = []
for label in set(labels):
    merged_bbox = None
    cluster_text = ""

    for i, (text, box) in enumerate(corrected_boxes):
        if labels[i] == label:
            cluster_text += text + " "
            merged_bbox = merge_boxes([merged_bbox, box]) if merged_bbox is not None else box

    if merged_bbox is not None:
        final_boxes.append((cluster_text.strip(), merged_bbox))

print(f"Merged boxes into {len(final_boxes)} final boxes.")

# Check for overlaps and merge those boxes
combined_boxes = []
for text, box in final_boxes:
    found_overlap = False
    for i, (combined_text, combined_box) in enumerate(combined_boxes):
        if boxes_overlap(box, combined_box):
            combined_text += ' ' + text
            combined_box = merge_boxes([combined_box, box])
            combined_boxes[i] = (combined_text.strip(), combined_box)  # Update existing entry
            found_overlap = True
            break
    if not found_overlap:
        combined_boxes.append((text, box))

print(f"Combined boxes into {len(combined_boxes)} boxes after overlap checking.")

# Draw final bounding boxes and check for expected phrases
print("Drawing final bounding boxes...")
for text, box in combined_boxes:
    top_left, top_right, bottom_right, bottom_left = box
    x_min = int(min(top_left[0], bottom_left[0]))
    y_min = int(min(top_left[1], top_right[1]))
    x_max = int(max(top_right[0], bottom_right[0]))
    y_max = int(max(bottom_left[1], bottom_right[1]))

    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

    # Check for expected phrases
    for expected in expected_phrases:
        if expected.lower() in text.lower():
            print(f"Expected phrase '{expected}' found in text: '{text}'")
            cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red text
            break  # Exit after first match

# Save the final image with highlighted matches
output_path = 'data/screenshots/final_result.png'  # Change this to your desired output path
cv2.imwrite(output_path, image)
print(f"Saved the final image with highlighted matches to {output_path}.")

# Print all extracted texts
print("Extracted Texts:")
for text, _ in combined_boxes:
    print(text)