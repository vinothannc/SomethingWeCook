import os
import io
from urllib.parse import urlparse

import requests
import numpy as np
from groq import Groq
from PIL import Image, UnidentifiedImageError
import csv
import easyocr
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Groq with Mixtral 8x7B
client = Groq(api_key=GROQ_API_KEY)


# Function to extract text from image using OCR (modified for URLs)
def extract_text_from_image_url(image_url):
    """Extracts text from an image using OCR, handling both URLs and local file paths."""

    if not image_url:
        print("Warning: Skipping empty image URL.")
        return ""

    parsed_url = urlparse(image_url)
    local_path = None

    try:
        if parsed_url.scheme in ["http", "https"]:
            #  Handle online images
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                print(f"Error: URL {image_url} does not contain an image. Content-Type: {content_type}")
                return ""

            image = Image.open(io.BytesIO(response.content))

        elif parsed_url.scheme == "file":
            #  Extract the correct local file path
            local_path = os.path.abspath(parsed_url.path)

            #  Windows Fix (Remove extra slashes in file path)
            if os.name == "nt":
                local_path = local_path.lstrip("\\")

            #  Ensure the file exists
            if not os.path.exists(local_path):
                print(f"Error: File does not exist - {local_path}")
                return ""

            #  Open the local file directly
            image = Image.open(local_path)

        else:
            print(f"Error: Unsupported image source - {image_url}")
            return ""

        #  Convert to numpy array and apply OCR
        image_array = np.array(image)
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(image_array, detail=0, paragraph=True)
        cleaned_text = ' '.join(result).strip()
        cleaned_text = re.sub(r'\b(copyright|all rights reserved|202[0-9])\b', '', cleaned_text, flags=re.IGNORECASE)

        print(f" Successfully extracted text from: {image_url}")
        return cleaned_text

    except (requests.exceptions.RequestException, UnidentifiedImageError, FileNotFoundError) as e:
        print(f" Error processing image: {image_url} - {e}")
        return ""

# Enhanced Prompt for Improved Detection
PROMPT_TEMPLATE = (
    "Analyze the following content and classify the website as either 'Parked Domain' or 'Functional Website'.\n\n"
    "Classification Criteria:\n"
    "- Parked Domain: Typically includes placeholder text such as 'This domain is for sale', 'Buy this domain', or references to domain registration services. "
    "Avoid classifying websites as 'Parked Domain' solely based on advertisements, as they can appear on functional websites as well. Focus on clear indicators of parked domains.\n"
    "- Functional Website: Includes legitimate content about products, services, or useful information. "
    "Websites that display error messages like '404 Not Found', 'Server Error', or 'This webpage is temporarily unavailable' should also be classified as 'Functional Website' unless there are strong indicators of a parked domain.\n\n"
    "Important Note: Provide only the classification result as either 'Parked Domain' or 'Functional Website' without additional explanation.\n\n"
    "Content: {text}"
)


def analyze_with_llm(extracted_text):
    if not extracted_text.strip():
        return "Functional Website"

    prompt = PROMPT_TEMPLATE.format(text=extracted_text)
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=5,
            top_p=0.9,
            stream=False,
            stop=None,
        )
        response_text = completion.choices[0].message.content.strip()
        print(response_text)
        return response_text
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        return "Functional Website"


def detect_parked_domain(image_urls):
    classifications = []

    for image_url in image_urls:
        text = extract_text_from_image_url(image_url)
        classification = analyze_with_llm(text)
        classifications.append(classification)

    # Determine final classification
    final_classification = "Parked Domain" if "Parked Domain" in classifications else "Functional Website"

    return final_classification

def process_images(domain_image_url):

    classification = detect_parked_domain(domain_image_url)

    return classification
