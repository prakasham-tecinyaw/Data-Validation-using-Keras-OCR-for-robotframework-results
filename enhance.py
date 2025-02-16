import re

# Define field-value mapping
field_value_map = {
    "Name": r"^[A-Za-z ]+$",
    "Address": r"^\d+\s[A-Za-z\s]+",
    "Date": r"^\d{4}-\d{2}-\d{2}$",
    "Amount": r"^\$?\d+(?:,\d{3})*(?:\.\d{2})?$"
}

# Example detected text from OCR
detected_texts = ["John Doe", "123 Main St", "2023-02-16", "$150.00"]

# Store extracted data
extracted_data = {}

# Match detected text to fields
for text in detected_texts:
    for field, pattern in field_value_map.items():
        if re.match(pattern, text):
            extracted_data[field] = text
            break

# Print the extracted data
print("Extracted Data:", extracted_data)