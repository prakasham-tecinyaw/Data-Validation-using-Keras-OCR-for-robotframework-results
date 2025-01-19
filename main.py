import os
import pandas as pd  # Assuming you're using pandas to read the CSV
import re  # For regex operations
from src.ocr_extractor import extract_text_with_keras_ocr
from src.logger import log_results

def normalize_text(text):
    """Normalize text by converting to lowercase, stripping whitespace, and removing punctuation."""
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Strip leading and trailing whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def read_csv_values(csv_path):
    """Reads the CSV file and returns a list of expected values."""
    df = pd.read_csv(csv_path)
    return df.iloc[:, 1].tolist()  # Get values from the second column

def validate_values(extracted_text, expected_values):
    """Check if each expected value is in the normalized extracted text."""
    normalized_extracted_text = normalize_text(extracted_text)
    results = []
    
    for expected_value in expected_values:
        normalized_expected_value = normalize_text(expected_value)
        
        if normalized_expected_value not in normalized_extracted_text:
            message = f"Value mismatch: expected '{expected_value}', but not found in extracted text."
            results.append(message)
        else:
            message = f"Value matched: '{expected_value}' found in extracted text."
            results.append(message)

    return results

def main(image_path, csv_path, log_file):
    # Check file existence
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} does not exist.")
        return

    try:
        extracted_text = extract_text_with_keras_ocr(image_path)

        expected_values = read_csv_values(csv_path)
        print(f"Expected values: {expected_values}")
        results = validate_values(extracted_text, expected_values)
        print(f"Validation results: {results}")
        
        # Prepare log entries
        log_entries = [f"Extracted text: '{extracted_text}'"]
        log_entries.extend(results)  # Add validation results

        # Log all entries (overwrite the log file)
        log_results(log_entries, log_file)
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    image_path = 'data/screenshots/screenshot.png'
    csv_path = 'data/expected_values.csv'
    log_file = 'logs/validation_log.txt'
    
    main(image_path, csv_path, log_file)