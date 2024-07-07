import pytesseract
from transformers import pipeline
from PIL import Image
import pandas as pd
import os

# Set up Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Update this to your Tesseract path if necessary

# Initialize the NLP pipeline for text classification
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Define categories to classify text
categories = ["title", "author", "publisher", "extra information"]

def extract_text(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error in extracting text: {e}")
        return ""

def classify_text(text):
    try:
        lines = text.split('\n')
        results = {"title": None, "author": None, "publisher": None, "extra information": []}
        
        for line in lines:
            if line.strip():
                result = classifier(line, categories)
                label = result['labels'][0]
                if label in ["title", "author", "publisher"]:
                    if not results[label]:
                        results[label] = line.strip()
                else:
                    results["extra information"].append(line.strip())
        
        return results
    except Exception as e:
        print(f"Error in classifying text: {e}")
        return {}

def save_to_excel(data, filename="books_info.xlsx"):
    try:
        if not os.path.exists(filename):
            df = pd.DataFrame(columns=["Title", "Author", "Publisher", "Extra Information"])
        else:
            df = pd.read_excel(filename)

        new_entry = {
            "Title": data["title"],
            "Author": data["author"],
            "Publisher": data["publisher"],
            "Extra Information": "; ".join(data["extra information"])
        }
        df = df.append(new_entry, ignore_index=True)
        df.to_excel(filename, index=False)
    except Exception as e:
        print(f"Error in saving to Excel: {e}")

def main():
    image_path = '/path/to/your/captured_image.png'  # Update this path
    print("Extracting text from image...")
    text = extract_text(image_path)
    if text:
        print("Classifying information...")
        classified_data = classify_text(text)
        print("Saving classified data to Excel...")
        save_to_excel(classified_data)
        print("Data saved to Excel successfully.")
    else:
        print("No text extracted from the image.")

if __name__ == "__main__":
    main()
