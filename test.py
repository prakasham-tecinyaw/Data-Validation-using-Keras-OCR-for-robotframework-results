import keras_ocr
import matplotlib.pyplot as plt
import cv2

# Initialize the Keras OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Load the image
image_path = 'data/screenshots/screenshot.png'  # Change this to your image path
image = keras_ocr.tools.read(image_path)

# Use the pipeline to recognize text
prediction_groups = pipeline.recognize([image])

# Create a copy of the image to draw on
image_with_boxes = image.copy()

# Draw the bounding boxes on the image
for predictions in prediction_groups:
    for text, box in predictions:
        # Convert box coordinates to integers
        box = box.astype(int)
        color = (0, 255, 0)  # Green color for the box
        # Draw the polygon for the bounding box
        for i in range(4):
            start_point = tuple(box[i])
            end_point = tuple(box[(i + 1) % 4])
            cv2.line(image_with_boxes, start_point, end_point, color, thickness=2)

# Convert the image from BGR to RGB format for displaying
image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

# Display the image with recognized text
plt.imshow(image_with_boxes)
plt.axis('off')  # Hide axes
plt.show()