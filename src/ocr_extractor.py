import keras_ocr

def extract_text_with_keras_ocr(image_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [keras_ocr.tools.read(image_path)]
    prediction_groups = pipeline.recognize(images)
    
    extracted_text = []
    for predictions in prediction_groups:
        for text, box in predictions:
            extracted_text.append(text)
    
    return ' '.join(extracted_text)