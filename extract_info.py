import fitz  # PyMuPDF
import os
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image
from collections import defaultdict
import pytesseract
import cv2
import pytesseract
from pytesseract import Output
# Set the path to the Tesseract executable for Linux (Google Colab)
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  
from typing import List, Dict
from openai_api import Openai_API
from model import Responder
import base64
from pathlib import Path
from typing import Union
import json
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'




class ExtractInformation:
    def __init__(self,pdf_path):
        self.pdf_path = pdf_path
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        
    def pdf_to_image(self,dpi=300):
        """
        Convert PDF pages to images and save them in the specified directory.
        """
        
        #make folder of pdf name to save images
        pdf_name = os.path.basename(self.pdf_path).split('.')[0]
        output_dir = os.path.join(os.getcwd(), pdf_name)
        os.makedirs(output_dir, exist_ok=True)
        # Open the PDF
        doc = fitz.open(self.pdf_path)
        # Loop through each page and save as image
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
            pix.save(image_path)
        print(f"Saved {len(doc)} pages as images in '{output_dir}' folder.")
        return output_dir
    
    def find_table_coordinates(self, folder, padding_ratio=0.05):
   

        table_info = defaultdict(list)
        file_paths = os.listdir(folder)

        for file_path in file_paths:
            image_path = os.path.join(folder, file_path)
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=0.9, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                label_text = self.model.config.id2label[label.item()]
                print(
                    f"Detected {label_text} with confidence {round(score.item(), 3)} at location {box}"
                )

                if box:
                    # Apply padding to the box
                    x_min, y_min, x_max, y_max = box

                    pad_x = (x_max - x_min) * padding_ratio
                    pad_y = (y_max - y_min) * padding_ratio

                    x_min = max(0, x_min - pad_x)
                    y_min = max(0, y_min - pad_y)
                    x_max = min(width, x_max + pad_x)
                    y_max = min(height, y_max + pad_y)

                    padded_box = [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)]
                    table_info[file_path].append(padded_box)

        return dict(table_info)
# Just one dict, easy to use
    
    def crop_table_region(self, image_path, bbox, output_path):
        """
        Crops the table region from the image based on the bounding box.

        Parameters:
        - image_path: Path to the original page image.
        - bbox: Bounding box coordinates [x0, y0, x1, y1].
        - output_path: Path to save the cropped table image.
        """
        image = Image.open(image_path)
        cropped_image = image.crop(bbox)
        cropped_image.save(output_path)
        print("image_saved")
        
    def extract_boxes_from_image(self,image_path: str) -> List[Dict]:
        """
        Extracts text and bounding boxes from the given image using Tesseract OCR.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            List[Dict]: A list of dictionaries with text, position, size, and confidence.
        """
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {image_path}")
            return []

        try:
            img = cv2.imread(image_path)
            if img is None:
                print("❌ Error: Image could not be loaded.")
                return []

            # Preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR with bounding boxes
            data = pytesseract.image_to_data(thresh, config='--oem 3 --psm 6', output_type=Output.DICT)

            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                if conf > 60 and text:
                    results.append({
                        "text": text,
                        "left": data['left'][i],
                        "top": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i],
                        "confidence": conf
                    })

            return results

        except Exception as e:
            print(f"❌ Exception occurred during OCR: {e}")
            return []

    def encode_image_to_base64(self,image_path: Union[str, Path]) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def extract_numerics(self, folder,image_path, text_data): 
        print("egeger")
        prompt = f"""
        Align the extracted OCR text with give Financial data table image. Analyze numeric or percentage data column wise.
        Make sure that you are not missing any information in the given table image.
        Return the output in JSON format.
        {{
        "<Detected Label>": {{
            "total": null,
            "components": [list of numeric values or percentages in that section]
        }}
        }}

    Return only the JSON output without extra commentary.

    Here is the extracted OCR text:
    {text_data}
    """
        image_path = os.path.join(folder, image_path)

        # Encode image
        base64_image = self.encode_image_to_base64(image_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ]
        }]

        openai_api = Openai_API()
        
        response: Responder = openai_api.generate_few_shot_text(
            messages,
            model="gpt-4o-mini",
            max_tokens=2000,
            temperature=0.1
        )

        if response.status == "success":
            return response.message

        return response.message  # Fallback on failure

    
    # def verify_totals(self, data):
    #     tolerance = 0.01  # small rounding tolerance for floating-point sums

    #     for section, values in data.items():
    #         total = values["total"]
    #         components = values["components"]
    #         #replace components null value with 0
    #         components = [float(x) if x is not None else 0 for x in components]
    #         sum_components = sum(components)
    #         print(sum_components)

    #         if total is None:
    #             print(f"{section}: Skipped (total is None)")
    #         else:
    #             if abs(sum_components - total) <= tolerance:
    #                 print(f"{section}: ✅ Valid (sum matches total)")
    #             else:
    #                 print(f"{section}: ❌ Invalid (sum = {sum_components:.2f}, total = {total:.2f})")
            
    def verify_totals(self, data):
        tolerance = 0.01  # small rounding tolerance
        results = []

        for section, values in data.items():
            total = values["total"]
            components = values["components"]
            components = [float(x) if x is not None else 0 for x in components]
            sum_components = sum(components)

            if total is None:
                results.append(f"**{section}**: ⚠️ Skipped (total is None)")
            else:
                if abs(sum_components - total) <= tolerance:
                    results.append(f"**{section}**: ✅ Valid (sum matches total)")
                else:
                    results.append(
                        f"**{section}**: ❌ Invalid (sum = {sum_components:.2f}, total = {total:.2f})"
                    )

        return results
            
    def remove_backticks(self,text):
        if text.startswith("```json"):
            text = text.replace("```json", "", 1)
        elif text.startswith("```"):
            text = text.replace("```", "", 1)
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def extract_text_from_image(self, folder, image_path):
      
        # img_cv = cv2.imread(image_path)

        # # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
        # # we need to convert from BGR to RGB format/mode:
        # img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # # print(pytesseract.image_to_string(img_rgb))
        # # OR
        # img_rgb = Image.frombytes('RGB', img_cv.shape[:2], img_cv, 'raw', 'BGR', 0, 0)
        # text = pytesseract.image_to_string(img_rgb)
     
        # Open the image file
        # Check if the file exists before trying to read it
        image_path = os.path.join(folder, image_path)
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return ""  # Return empty string if file not found
        
        img = cv2.imread(image_path)
        
        # Convert the image to grayscale (improves OCR accuracy)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Optional: Apply thresholding for better accuracy
        _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

        # Use Tesseract to extract text
        text = pytesseract.image_to_string(thresh_img, config='--psm 6')  # Use page segmentation mode 6 for OCR on a table

        return text
      
# Example usage
        

# pdf_path = "data/2024-05-08-lt-financial-results-for-the-year-ended-march-31-2024.pdf"
# pdf_name = os.path.basename(pdf_path).split('.')[0]
# table_info = ExtractInformation(pdf_path=pdf_path)
# output_dir = table_info.pdf_to_image(dpi=300)
# print(output_dir) #directory where images are saved


# table_coordinates = table_info.find_table_coordinates(output_dir)
# cropped_images_folder =  f"cropped_images_{pdf_name}" #directory name
# os.makedirs(cropped_images_folder, exist_ok=True)
# for page, coordinates in table_coordinates.items():
#     folder = output_dir
#     complete_image_path = os.path.join(output_dir,page)
#     for cordinate in coordinates:
#       output_path = os.path.join(cropped_images_folder, f"{page}_{cordinate}.png")
#       table_info.crop_table_region(complete_image_path, cordinate, output_path)

# cropped_images = os.listdir(cropped_images_folder)

 
# image = "page_8.png_[148.2, 1363.28, 2364.69, 2682.44].png"

# text_data= table_info.extract_text_from_image(cropped_images_folder, image)
# print("ocr data",text_data)
# results= table_info.extract_numerics(cropped_images_folder, image,text_data)
# print(results)
# results = table_info.remove_backticks(results)
# print(results)


# results = json.loads(results)
  
# final_results = table_info.verify_totals(results)
# print(final_results)
   

