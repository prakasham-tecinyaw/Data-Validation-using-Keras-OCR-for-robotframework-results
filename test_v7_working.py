import cv2
import easyocr

# Load the image and perform OCR
image_path = 'data/screenshots/screenshot.png'  # Change this to your image path
image = cv2.imread(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])
# reader = easyocr.Reader(['en'], detect_network = 'dbnet18')
results = reader.readtext(image)

# Define expected phrases to check for redrawing boxes
expected_phrases = ["Groceries & Pets", "Watches", "Automotive", "Women's Bags", "Fashion"]

# Function to check if two bounding boxes overlap
def boxes_overlap(box1, box2):
    x1_min = min(box1[0][0], box1[2][0])
    x1_max = max(box1[0][0], box1[2][0])
    y1_min = min(box1[0][1], box1[2][1])
    y1_max = max(box1[0][1], box1[2][1])

    x2_min = min(box2[0][0], box2[2][0])
    x2_max = max(box2[0][0], box2[2][0])
    y2_min = min(box2[0][1], box2[2][1])
    y2_max = max(box2[0][1], box2[2][1])

    return (x1_min < x2_max and x1_max > x2_min and
            y1_min < y2_max and y1_max > y2_min)

# Function to merge overlapping bounding boxes with an increased clustering ratio
def merge_bounding_boxes(extracted_texts):
    merged_boxes = []
    used_indices = set()

    for i, (bbox_i, text_i, _) in enumerate(extracted_texts):
        if i in used_indices:
            continue
        
        # Initialize merged bounding box and text
        merged_bbox = list(bbox_i)
        merged_text = text_i

        for j, (bbox_j, text_j, _) in enumerate(extracted_texts):
            if i != j and j not in used_indices:
                if boxes_overlap(bbox_i, bbox_j):
                    # Update merged bounding box coordinates
                    merged_bbox[0] = (min(merged_bbox[0][0], bbox_j[0][0]), min(merged_bbox[0][1], bbox_j[0][1]))
                    merged_bbox[1] = (max(merged_bbox[1][0], bbox_j[1][0]), merged_bbox[1][1])
                    merged_bbox[2] = (max(merged_bbox[2][0], bbox_j[2][0]), max(merged_bbox[2][1], bbox_j[2][1]))
                    merged_bbox[3] = (min(merged_bbox[3][0], bbox_j[3][0]), min(merged_bbox[3][1], bbox_j[3][1]))
                    merged_text += ' ' + text_j
                    used_indices.add(j)

        merged_boxes.append((merged_text.strip(), tuple(merged_bbox)))
        used_indices.add(i)

    return merged_boxes

# Function to cluster nearby bounding boxes with increased clustering ratio
def cluster_nearby_bounding_boxes(extracted_texts, proximity_threshold=10):
    clustered_boxes = []
    used_indices = set()

    for i, (text_i, bbox_i) in enumerate(extracted_texts):
        if i in used_indices:
            continue
        
        merged_bbox = list(bbox_i)
        merged_text = text_i

        for j, (text_j, bbox_j) in enumerate(extracted_texts):
            if i != j and j not in used_indices:
                # Check proximity in both dimensions
                if (abs(bbox_i[0][0] - bbox_j[0][0]) < proximity_threshold and
                    abs(bbox_i[0][1] - bbox_j[0][1]) < proximity_threshold):
                    # Update merged bounding box coordinates
                    merged_bbox[0] = (min(merged_bbox[0][0], bbox_j[0][0]), min(merged_bbox[0][1], bbox_j[0][1]))
                    merged_bbox[1] = (max(merged_bbox[1][0], bbox_j[1][0]), merged_bbox[1][1])
                    merged_bbox[2] = (max(merged_bbox[2][0], bbox_j[2][0]), max(merged_bbox[2][1], bbox_j[2][1]))
                    merged_bbox[3] = (min(merged_bbox[3][0], bbox_j[3][0]), min(merged_bbox[3][1], bbox_j[3][1]))
                    merged_text += ' ' + text_j
                    used_indices.add(j)

        clustered_boxes.append((merged_text.strip(), tuple(merged_bbox)))
        used_indices.add(i)

    return clustered_boxes

# Merge overlapping bounding boxes
merged_texts = merge_bounding_boxes(results)

# Cluster nearby bounding boxes
clustered_texts = cluster_nearby_bounding_boxes(merged_texts)

# Draw initial bounding boxes and extract text
for (text, bbox) in clustered_texts:
    top_left, top_right, bottom_right, bottom_left = bbox
    x_min = int(min(top_left[0], bottom_left[0]))
    y_min = int(min(top_left[1], top_right[1]))
    x_max = int(max(top_right[0], bottom_right[0]))
    y_max = int(max(bottom_left[1], bottom_right[1]))

    # Draw initial bounding box in red
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red box for all text

# Check the extracted text against expected phrases and redraw boxes for matches
for (text, bbox) in clustered_texts:
    for phrase in expected_phrases:
        if phrase in text:
            top_left, top_right, bottom_right, bottom_left = bbox
            x_min = int(min(top_left[0], bottom_left[0]))
            y_min = int(min(top_left[1], top_right[1]))
            x_max = int(max(top_right[0], bottom_right[0]))
            y_max = int(max(bottom_left[1], bottom_right[1]))

            # Draw bounding box in green for matched phrases
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box for matched phrases

# Save the final image with highlighted matches
output_path = 'data/screenshots/final_result.png'  # Change this to your desired output path
cv2.imwrite(output_path, image)

# Print all extracted texts and matching expected phrases
print("Extracted Texts:")
for text, _ in clustered_texts:
    print(text)

print(f"Saved the image with highlighted matches to {output_path}.")