from django.test import TestCase

# Create your tests here.


import io
import time
import requests
import numpy as np
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import cloudinary
import cloudinary.uploader
from ultralytics import YOLO
import cv2
import easyocr
from fuzzywuzzy import fuzz
import cloudinary.uploader


# Cloudinary Configuration
cloudinary.config(
    cloud_name="dxtwdkodr",  # Replace with your Cloudinary cloud name
    api_key="252621686232783",  # Replace with your Cloudinary API key
    api_secret="jOvEJWImDaRidWoFhHLWcoYfLRw"  # Replace with your Cloudinary API secret
)

# Selenium WebDriver Setup
options = Options()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--start-maximized")  # Start browser in full-screen
options.add_argument("--ignore-certificate-errors")
options.add_argument("--log-level=3")
options.add_argument("--inprivate")  # Use incognito mode
service = Service(EdgeChromiumDriverManager().install())
driver = webdriver.Edge(service=service, options=options)


# Screenshot Capture & Upload
def scroll_and_capture_dtdc(url):  # DTDC
    """
    Scrolls through a webpage at fixed intervals, captures up to 5 screenshots,
    and stores them under the specified folder 'DTDC' in Cloudinary.
    """
    print(f"📸 Capturing screenshots for: {url}")
    cloudinary_urls = []

    # Add "https://www." prefix if missing
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://www." + url

    try:
        # Set a timeout for page load
        driver.set_page_load_timeout(40)
        driver.get(url)  # Navigate to the URL

    except Exception as e:
        print(f"⚠️ Page load timeout or error: {e}")
        # Capture a screenshot of whatever content is available
        screenshot_bytes = driver.get_screenshot_as_png()
        image_stream = io.BytesIO(screenshot_bytes)

        # Specify the folder name
        folder_name = "DTDC"

        public_id = f"{folder_name}/{url.split('//')[-1].replace('/', '_')}_timeout"
        try:
            upload_response = cloudinary.uploader.upload(image_stream, public_id=public_id, resource_type="image")
            cloud_url = upload_response.get("secure_url")
            print(f"✅ Partial screenshot uploaded to Cloudinary folder '{folder_name}': {cloud_url}")
            cloudinary_urls.append(cloud_url)
        except Exception as upload_error:
            print(f"❌ Failed to upload partial screenshot: {upload_error}")
        return cloudinary_urls  # Return immediately after capturing a screenshot in case of timeout

    # Normal scrolling and capturing logic
    y_offset = 0
    scroll_step = 1000  # Fixed scroll step in pixels
    screenshot_count = 1
    max_screenshots = 4  # Limit to a maximum of 5 screenshots

    while screenshot_count <= max_screenshots:
        driver.execute_script(f"window.scrollTo(0, {y_offset});")
        time.sleep(2)  # Allow time for scroll effect

        # Capture screenshot as PNG bytes
        screenshot_bytes = driver.get_screenshot_as_png()
        image_stream = io.BytesIO(screenshot_bytes)

        # Specify the folder name
        folder_name = "DTDC"

        # Create a public ID with folder structure
        public_id = f"{folder_name}/{url.split('//')[-1].replace('/', '_')}_part{screenshot_count}"

        try:
            upload_response = cloudinary.uploader.upload(image_stream, public_id=public_id, resource_type="image")
            cloud_url = upload_response.get("secure_url")
            cloudinary_urls.append(cloud_url)
            print(f"✅ Screenshot uploaded to Cloudinary folder '{folder_name}': {cloud_url}")
        except Exception as e:
            print(f"❌ Cloudinary upload failed for part {screenshot_count}: {e}")
            break

        # Check if reached the bottom of the page
        last_height = driver.execute_script("return document.body.scrollHeight")
        if y_offset >= last_height:
            break

        # Move to the next scroll position
        y_offset += scroll_step
        screenshot_count += 1

    return cloudinary_urls


# Screenshot Capture & Upload
def scroll_and_capture_aliceblue(url):  # aliceblue
    """
    Scrolls through a webpage at fixed intervals, captures up to 5 screenshots,
    and stores them under the specified folder 'DTDC' in Cloudinary.
    """
    print(f"📸 Capturing screenshots for: {url}")
    cloudinary_urls = []

    # Add "https://www." prefix if missing
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://www." + url

    try:
        # Set a timeout for page load
        driver.set_page_load_timeout(40)
        driver.get(url)  # Navigate to the URL

    except Exception as e:
        print(f"⚠️ Page load timeout or error: {e}")
        # Capture a screenshot of whatever content is available
        screenshot_bytes = driver.get_screenshot_as_png()
        image_stream = io.BytesIO(screenshot_bytes)

        # Specify the folder name
        folder_name = "aliceblue"

        public_id = f"{folder_name}/{url.split('//')[-1].replace('/', '_')}_timeout"
        try:
            upload_response = cloudinary.uploader.upload(image_stream, public_id=public_id, resource_type="image")
            cloud_url = upload_response.get("secure_url")
            print(f"✅ Partial screenshot uploaded to Cloudinary folder '{folder_name}': {cloud_url}")
            cloudinary_urls.append(cloud_url)
        except Exception as upload_error:
            print(f"❌ Failed to upload partial screenshot: {upload_error}")
        return cloudinary_urls  # Return immediately after capturing a screenshot in case of timeout

    # Normal scrolling and capturing logic
    y_offset = 0
    scroll_step = 1000  # Fixed scroll step in pixels
    screenshot_count = 1
    max_screenshots = 4  # Limit to a maximum of 5 screenshots

    while screenshot_count <= max_screenshots:
        driver.execute_script(f"window.scrollTo(0, {y_offset});")
        time.sleep(2)  # Allow time for scroll effect

        # Capture screenshot as PNG bytes
        screenshot_bytes = driver.get_screenshot_as_png()
        image_stream = io.BytesIO(screenshot_bytes)

        # Specify the folder name
        folder_name = "aliceblue"

        # Create a public ID with folder structure
        public_id = f"{folder_name}/{url.split('//')[-1].replace('/', '_')}_part{screenshot_count}"

        try:
            upload_response = cloudinary.uploader.upload(image_stream, public_id=public_id, resource_type="image")
            cloud_url = upload_response.get("secure_url")
            cloudinary_urls.append(cloud_url)
            print(f"✅ Screenshot uploaded to Cloudinary folder '{folder_name}': {cloud_url}")
        except Exception as e:
            print(f"❌ Cloudinary upload failed for part {screenshot_count}: {e}")
            break

        # Check if reached the bottom of the page
        last_height = driver.execute_script("return document.body.scrollHeight")
        if y_offset >= last_height:
            break

        # Move to the next scroll position
        y_offset += scroll_step
        screenshot_count += 1

    return cloudinary_urls


# Screenshot Capture & Upload
def scroll_and_capture_puravankara(url):  # Puravankara
    """
    Scrolls through a webpage at fixed intervals, captures up to 5 screenshots,
    and stores them under the specified folder 'DTDC' in Cloudinary.
    """
    print(f"📸 Capturing screenshots for: {url}")
    cloudinary_urls = []

    # Add "https://www." prefix if missing
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://www." + url

    try:
        # Set a timeout for page load
        driver.set_page_load_timeout(40)
        driver.get(url)  # Navigate to the URL

    except Exception as e:
        print(f"⚠️ Page load timeout or error: {e}")
        # Capture a screenshot of whatever content is available
        screenshot_bytes = driver.get_screenshot_as_png()
        image_stream = io.BytesIO(screenshot_bytes)

        # Specify the folder name
        folder_name = "Puravankara"

        public_id = f"{folder_name}/{url.split('//')[-1].replace('/', '_')}_timeout"
        try:
            upload_response = cloudinary.uploader.upload(image_stream, public_id=public_id, resource_type="image")
            cloud_url = upload_response.get("secure_url")
            print(f"✅ Partial screenshot uploaded to Cloudinary folder '{folder_name}': {cloud_url}")
            cloudinary_urls.append(cloud_url)
        except Exception as upload_error:
            print(f"❌ Failed to upload partial screenshot: {upload_error}")
        return cloudinary_urls  # Return immediately after capturing a screenshot in case of timeout

    # Normal scrolling and capturing logic
    y_offset = 0
    scroll_step = 1000  # Fixed scroll step in pixels
    screenshot_count = 1
    max_screenshots = 4  # Limit to a maximum of 5 screenshots

    while screenshot_count <= max_screenshots:
        driver.execute_script(f"window.scrollTo(0, {y_offset});")
        time.sleep(2)  # Allow time for scroll effect

        # Capture screenshot as PNG bytes
        screenshot_bytes = driver.get_screenshot_as_png()
        image_stream = io.BytesIO(screenshot_bytes)

        # Specify the folder name
        folder_name = "Puravankara"

        # Create a public ID with folder structure
        public_id = f"{folder_name}/{url.split('//')[-1].replace('/', '_')}_part{screenshot_count}"

        try:
            upload_response = cloudinary.uploader.upload(image_stream, public_id=public_id, resource_type="image")
            cloud_url = upload_response.get("secure_url")
            cloudinary_urls.append(cloud_url)
            print(f"✅ Screenshot uploaded to Cloudinary folder '{folder_name}': {cloud_url}")
        except Exception as e:
            print(f"❌ Cloudinary upload failed for part {screenshot_count}: {e}")
            break

        # Check if reached the bottom of the page
        last_height = driver.execute_script("return document.body.scrollHeight")
        if y_offset >= last_height:
            break

        # Move to the next scroll position
        y_offset += scroll_step
        screenshot_count += 1

    return cloudinary_urls


# ocr and similarity check
ocr_reader = easyocr.Reader(['en'])


def extract_text(image_path):
    """Extract text using EasyOCR."""
    results = ocr_reader.readtext(image_path)
    extracted_texts = [result[1].strip().upper() for result in results]
    return extracted_texts


def check_similarity(extracted_texts, keywords):
    """Check similarity between extracted texts and keywords."""
    for text in extracted_texts:
        for keyword in keywords:
            if fuzz.ratio(text, keyword) > 70:  # 70% similarity threshold
                return True, text
    return False, None


# orb and shift similarity check
def download_image(image_url):
    """Download an image from a URL and return it as a NumPy array."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Failed to load image from URL: {image_url}")
            return None
        return img
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None


def orb_similarity(ref_img, cropped_img):
    """Compute ORB similarity score between two images."""
    ref_img = cv2.resize(ref_img, (768, 768))
    cropped_img = cv2.resize(cropped_img, (768, 768))
    ref_img = cv2.GaussianBlur(ref_img, (5, 5), 0)
    cropped_img = cv2.GaussianBlur(cropped_img, (5, 5), 0)

    orb = cv2.ORB_create(nfeatures=800)
    keypoints1, descriptors1 = orb.detectAndCompute(ref_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(cropped_img, None)

    if descriptors1 is None or descriptors2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    good_matches = [m for m in matches if m.distance < 70]
    similarity_score = len(good_matches) / max(len(keypoints1), len(keypoints2))
    return similarity_score


def sift_similarity(ref_img, cropped_img):
    """Compute SIFT similarity score between two images."""
    ref_img = cv2.resize(ref_img, (768, 768))
    cropped_img = cv2.resize(cropped_img, (768, 768))
    ref_img = cv2.GaussianBlur(ref_img, (5, 5), 0)
    cropped_img = cv2.GaussianBlur(cropped_img, (5, 5), 0)

    sift = cv2.SIFT_create(nfeatures=800)
    keypoints1, descriptors1 = sift.detectAndCompute(ref_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(cropped_img, None)

    if descriptors1 is None or descriptors2 is None:
        return 0

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    similarity_score = len(good_matches) / max(len(keypoints1), len(keypoints2))
    return similarity_score


reference_image_urls_dtdc = [
    "https://res.cloudinary.com/dxtwdkodr/image/upload/v1742889470/k4q1an5lfxqzdyzhtmir.jpg",
    "https://res.cloudinary.com/dxtwdkodr/image/upload/v1742889471/ltxwi2czv62q4xblrfuo.jpg",
    "https://res.cloudinary.com/dxtwdkodr/image/upload/v1742889472/hkqguzl1cptk2wcvqkhf.jpg",
    "https://res.cloudinary.com/dxtwdkodr/image/upload/v1742889473/l2clzmlb2mjcub9vbl3l.png",
    "https://res.cloudinary.com/dxtwdkodr/image/upload/v1742964673/qfcag5alpg53u6unacyu.jpg"
]

# Initialize YOLO Models
model_path_purvankara = r"/home/vinothan/Ai_models/scripts/models/provinkra_finetuned_2.pt"
yolo_model_purva = YOLO(model_path_purvankara)  # Purvankara

model_path_aliceblue = r"/home/vinothan/Ai_models/scripts/models/aliceblue_finetuned_2.pt"
yolo_model_aliceblue = YOLO(model_path_aliceblue)  # aliceblue

model_path_dtdc = r"/home/vinothan/Ai_models/scripts/models/dtdc_finetuned_part2.pt"
yolo_model_dtdc = YOLO(model_path_dtdc)  # Dtdc

# Define Keywords for Matching
keywords_puravankara = [
    "Puravankara", "Purvaankara", "Pura Vankara", "Pura-Vankara", "Pura.Vankara",
    "Puravankra", "Purvaankra", "Puravan Kara", "Purava Nkara", "P Vankara",
    "Pura V", "Puravankar", "Purvankara", "Puraankara", "Purvankra",
    "Pura Vankra", "PuravaNkra", "Purvankr", "P'Vankara", "Puravankaraa",
    "Purva", "P U R V A", "P.U.R.V.A", "P U R V", "Pur V",
    "P-Urva", "Purvaa", "Pur'va", "Purvaaa", "P/URVA", "Purva-land", "Purvaland", "Provident"
]  # purvankara
keywords_puravankara = [keyword.upper() for keyword in
                        keywords_puravankara]  # Uppercase for case-insensitive comparison

keywords_aliceblue = [
    "aliceblue", "aliceblue trading", "aliceblue broker", "aliceblue demat account",
    "aliceblue app", "aliceblue online trading", "aliceblue ANT", "aliceblue ANT trading platform",
    "aliceblue ANT web", "aliceblue ANT mobile app", "aliceblue ANT desk", "aliceblue trading API",
    "aliceblue support", "aliceblue login", "aliceblue account opening", "aliceblue customer care"
]  # aliceblue
keywords_aliceblue = [keyword.upper() for keyword in keywords_aliceblue]

keywords_dtdc = [
    "DTDC", "D T D C", "D-T-D-C", "D.T.D.C", "DTDCX", "DTCC", "DTD", "D T D",
    "D-T-D", "D.T.D", "DTC", "TDC", "D T C", "T D C", "D T D O", "DTD0",
    "DTDQ", "DTDO", "D T D Q", "DTDC0", "DTDCQ", "DTIC", "DTLC", "D I D C",
    "D T I C", "D T L C", "TD DC", "DC TD", "D C T D", "T D D C", "T D C D",
    "C D T D", "C T D D", "C D T C", "T D T C", "T T D C", "D_T_D_C", "D TDC",
    "D T-DC", "D-T DC", "DT-DC", "D_T_DC", "DTDC Express", "DTDC Courier",
    "DTDC Logistics", "DTDC India", "DTDC Pvt Ltd", "DTDC Limited", "DTDC Service",
    "DTDC Online", "DT D C", "D T D C X", "DT-DCX", "DTD C", "DTD Express",
    "D T D C E", "D T C E", "DT DCE", "D TD C", "DLD"
]  # dtdc
keywords_dtdc = [keyword.upper() for keyword in keywords_dtdc]


# Phishing Detection Pipeline Purvankara
def process_image_url_puravankara(image_url):
    print(f"🔍 Processing image URL: {image_url}")

    # Step 1: Download the image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Failed to load image from URL: {image_url}")
            return False
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return False

    # Step 2: YOLO Detection
    results = yolo_model_purva(img)
    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            cls = int(box.cls[0])  # Get the class index
            conf = float(box.conf[0])  # Confidence score

            if names[cls] == "logo" and conf >= 0.5:
                # Step 3: Crop the detected region
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = img[y1:y2, x1:x2]

                # Step 4: Perform OCR on the cropped region
                rgb_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                extracted_texts = extract_text(rgb_image)
                is_similar, matched_text = check_similarity(extracted_texts, keywords_puravankara)
                if is_similar:
                    print(f"✅ Phishing keyword detected: '{matched_text}'")
                    return f"{image_url}: Phishing site"

    return f"{image_url}: Not a phishing site"


# Phishing Detection Pipeline aliceblue
def process_image_url_aliceblue(image_url):
    print(f"🔍 Processing image URL: {image_url}")

    # Step 1: Download the image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Failed to load image from URL: {image_url}")
            return False
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return False

    # Step 2: YOLO Detection
    results = yolo_model_aliceblue(img)
    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            cls = int(box.cls[0])  # Get the class index
            conf = float(box.conf[0])  # Confidence score

            if names[cls] == "logo" and conf >= 0.5:
                # Step 3: Crop the detected region
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = img[y1:y2, x1:x2]

                # Step 4: Perform OCR on the cropped region
                rgb_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                extracted_texts = extract_text(rgb_image)
                is_similar, matched_text = check_similarity(extracted_texts, keywords_aliceblue)
                if is_similar:
                    print(f"✅ Phishing keyword detected: '{matched_text}'")
                    return f"{image_url}: Phishing site"

    return f"{image_url}:Not a phishing site"


# Phishing Detection Pipeline DTDC
def process_image_url_dtdc(image_url, reference_image_urls):
    """
    Process an image URL to detect logos and check for similarity.
    Stops further processing once phishing is detected.
    """
    print(f"🔍 Processing image URL: {image_url}")

    # Step 1: Download the image
    img = download_image(image_url)
    if img is None:
        return f"{image_url}: Not a phishing site"

    # Step 2: YOLO Detection
    results = yolo_model_dtdc(img)
    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if names[cls] == "logo" and conf >= 0.5:  # Threshold for detection
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cropped_img = img[y1:y2, x1:x2]

                # Step 3: Perform OCR and check keyword similarity
                rgb_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                extracted_texts = extract_text(rgb_image)
                is_similar, matched_text = check_similarity(extracted_texts, keywords_dtdc)
                if is_similar:
                    print(f"✅ Phishing keyword detected: '{matched_text}'")
                    return f"{image_url}: Phishing site"

                # Step 4: If no keywords matched, check image similarity
                print(f"🔍 No keyword match. Proceeding to ORB/SIFT similarity checks...")
                for ref_img_url in reference_image_urls:
                    ref_img = download_image(ref_img_url)
                    if ref_img is None:
                        continue  # Skip if reference image could not be downloaded
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    cropped_img_gray = cv2.cvtColor(cropped_img,
                                                    cv2.COLOR_BGR2GRAY)  # Convert cropped image to grayscale

                    # Perform ORB similarity check
                    orb_score = orb_similarity(ref_img, cropped_img_gray)
                    if orb_score > 0.3:
                        print(f"✅ Similar image detected via ORB with score: {orb_score:.2f}")
                        return f"{image_url}: Phishing site"

                    sift_score = sift_similarity(ref_img, cropped_img_gray)
                    if sift_score > 0.3:
                        print(f"✅ Similar image detected via SIFT with score: {sift_score:.2f}")
                        return f"{image_url}: Phishing site"

    # If no matches found through OCR, ORB, or SIFT, return as not phishing
    return f"{image_url}: Not a phishing site"


# process pipeline

def process_pipeline_dtdc(url):
    cloudinary_urls = scroll_and_capture_dtdc(url)
    first_image_url = cloudinary_urls[0] if cloudinary_urls else None

    if first_image_url:
        # Iterate over all captured screenshots
        for image_url in cloudinary_urls:
            result = process_image_url_dtdc(image_url, reference_image_urls_dtdc)
            if "Phishing site" in result:  # Stop processing if phishing is detected
                return "Phising site"


        return "Not Phising site"


def process_pipeline_aliceblue(url):
    cloudinary_urls = scroll_and_capture_aliceblue(url)
    first_image_url = cloudinary_urls[0] if cloudinary_urls else None

    if first_image_url:
        # Iterate over all captured screenshots
        for image_url in cloudinary_urls:
            result = process_image_url_aliceblue(image_url)
            if "Phishing site" in result:  # Stop processing if phishing is detected
                return "Phising site"


        return "Not Phising site"


def process_pipeline_puravankara(url):
    cloudinary_urls = scroll_and_capture_puravankara(url)
    first_image_url = cloudinary_urls[0] if cloudinary_urls else None

    if first_image_url:
        # Iterate over all captured screenshots
        for image_url in cloudinary_urls:
            result = process_image_url_puravankara(image_url)
            if "Phishing site" in result:  # Stop processing if phishing is detected
                return "Phishing site"



        return "Not Phising site"


def route_pipeline(url, keyword):
    # Predefined original sites
    original_sites = ["dtdc.com", "dtdc.in", "dtdc.ai", "mydtdc.in", "dtdcbazaar.in",

                      "puravankara.com", "purvaland.com",

                      "aliceblueonline.com", "aliceblueindia.com"]

    # Split the input into URL and keyword
    try:
        url, keyword = url, keyword
    except ValueError:
        print("Invalid input format. Please provide in 'url:keyword' format.")
        return

    # Check if URL matches an original site
    if url in original_sites:
        return 'not phishing site'  # Exit the function as no pipeline needs to run for original sites

    # Otherwise, proceed with routing logic
    if keyword == "DTDC":
        print("Routing to DTDC pipeline...")
        status = process_pipeline_dtdc(url)
    elif keyword == "AliceBlue":
        print("Routing to AliceBlue pipeline...")
        status = process_pipeline_aliceblue(url)
    elif keyword == "Puravankara":
        print("Routing to Purvankara pipeline...")
        status = process_pipeline_puravankara(url)
    else:
        status = None

    return status


if __name__ == '__main__':
    # Example input
    final_status = route_pipeline(url="aliceblueonline.com", keyword="aliceblue")

    print(f"here is the final status : {final_status}")





