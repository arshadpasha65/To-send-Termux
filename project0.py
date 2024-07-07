import pytesseract
from transformers import pipeline
from PIL import Image
import pandas as pd
import os

# Initialize the NLP pipeline for text classification
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Define categories to classify text
categories = ["title", "author", "publisher", "extra information"]

def extract_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def classify_text(text):
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

def save_to_excel(data, filename="books_info.xlsx"):
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

def main():
    image_path = '/path/to/your/captured_image.png'  # Update this path
    print("Image captured, extracting text...")
    text = extract_text(image_path)
    print("Text extracted, classifying information...")
    classified_data = classify_text(text)
    print("Classification done, saving to Excel...")
    save_to_excel(classified_data)
    print("Data saved to Excel successfully.")

if __name__ == "__main__":
    main()