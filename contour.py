import cv2

# Load image
image = cv2.imread('form_image.png', cv2.IMREAD_GRAYSCALE)

# Preprocess image
_, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours to check for checkbox shapes
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50:  # Example area threshold
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        # Check if it resembles a checkbox (e.g., square shape)
        if 0.8 < aspect_ratio < 1.2:
            # Further logic to determine if checked or unchecked
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show detected checkboxes
cv2.imshow('Detected Checkboxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()