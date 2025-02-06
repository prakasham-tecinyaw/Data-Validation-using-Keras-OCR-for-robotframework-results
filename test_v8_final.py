import cv2
import keras_ocr
import os
import numpy as np
from sklearn.cluster import KMeans

# Set environment variable to force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the image
image_path = 'data/screenshots/image.png'  # Change this to your image path
image = cv2.imread(image_path)

# Initialize the Keras-OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Perform OCR using Keras-OCR
images = [image]
prediction_groups = pipeline.recognize(images)

# Expected phrases to look for
expected_phrases = ["Groceries & Pets", "Watches", "Automotive", "Women's Bags", "Fashion"]  # Modify this list as needed

# Function to calculate the center of a bounding box
def get_box_center(box):
    return (box[0][0] + box[2][0]) / 2, (box[0][1] + box[2][1]) / 2

# Function to check if two bounding boxes overlap
def boxes_overlap(box1, box2, threshold=5):
    x1_min = min(box1[0][0], box1[2][0])
    x1_max = max(box1[0][0], box1[2][0])
    y1_min = min(box1[0][1], box1[2][1])
    y1_max = max(box1[0][1], box1[2][1])

    x2_min = min(box2[0][0], box2[2][0])
    x2_max = max(box2[0][0], box2[2][0])
    y2_min = min(box2[0][1], box2[2][1])
    y2_max = max(box2[0][1], box2[2][1])

    return (x1_max > x2_min and x1_min < x2_max and
            y1_max > y2_min and y1_min < y2_max)

# Function to merge bounding boxes
def merge_boxes(boxes):
    min_x = min([min(box[0][0], box[2][0]) for box in boxes])
    min_y = min([min(box[0][1], box[2][1]) for box in boxes])
    max_x = max([max(box[1][0], box[3][0]) for box in boxes])
    max_y = max([max(box[2][1], box[3][1]) for box in boxes])

    return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

# Group boxes based on K-Means clustering
def cluster_boxes_kmeans(grouped_boxes, n_clusters):
    centers = np.array([get_box_center(box) for _, box in grouped_boxes])
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(centers)
    return kmeans.labels_

# Group boxes based on predictions
grouped_boxes = [(text, box) for text, box in prediction_groups[0] if text.strip()]

# Determine number of clusters (can be adjusted)
n_clusters = max(1, len(grouped_boxes) // 2)  # Ensure at least one cluster

# Apply K-Means clustering
labels = cluster_boxes_kmeans(grouped_boxes, n_clusters)

# Draw boxes for all detected clusters
final_boxes = []
for label in set(labels):
    merged_bbox = None
    cluster_text = ""

    for i, (text, box) in enumerate(grouped_boxes):
        if labels[i] == label:
            cluster_text += text + " "
            merged_bbox = merge_boxes([merged_bbox, box]) if merged_bbox is not None else box

    # Add the initial merged bounding box to the final boxes list
    if merged_bbox is not None:
        final_boxes.append((cluster_text.strip(), merged_bbox))

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

# Draw final bounding boxes and check for expected phrases
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
            cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red text
            break  # Exit after first match

# Save the final image with highlighted matches
output_path = 'data/screenshots/final_result.png'  # Change this to your desired output path
cv2.imwrite(output_path, image)

# Print all extracted texts
print("Extracted Texts:")
for text, _ in combined_boxes:
    print(text)

print(f"Saved the image with highlighted matches to {output_path}.")