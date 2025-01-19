import re  # For regex operations

def normalize_text(text):
    """Normalize text by converting to lowercase, stripping whitespace, and removing punctuation."""
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Strip leading and trailing whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def validate_values(extracted_text, expected_values):
    """Check if each expected value is in the extracted text."""
    results = []
    for expected_value in expected_values:
        if expected_value.strip().lower() not in extracted_text.strip().lower():
            message = f"Value mismatch: expected '{expected_value}', but not found in extracted text."
            results.append(message)
            print(message)  # Debugging line to show mismatches
    return results