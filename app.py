import streamlit as st
import os
import json
from extract_info import ExtractInformation  # Assuming your class is saved in extract_module.py
import shutil

st.set_page_config(page_title="Financial Table Extractor", layout="wide")
st.title("📊 Financial Table Extractor from PDF")

if "extractor" not in st.session_state:
    st.session_state.extractor = None
if "image_dir" not in st.session_state:
    st.session_state.image_dir = None
if "cropped_dir" not in st.session_state:
    st.session_state.cropped_dir = None
if "table_coords" not in st.session_state:
    st.session_state.table_coords = None

# Step 1: Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    pdf_path = os.path.join("uploads", uploaded_pdf.name)
    os.makedirs("uploads", exist_ok=True)
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    st.success("PDF uploaded successfully.")

    # Step 2: Convert PDF to images
    extractor = ExtractInformation(pdf_path)
    image_dir = extractor.pdf_to_image(dpi=300)
    
    st.session_state.extractor = extractor
    st.session_state.image_dir = image_dir

    st.success("PDF converted to images.")

    # Step 3: Detect tables
    with st.spinner("Detecting tables..."):
        table_coords = extractor.find_table_coordinates(image_dir)
    st.session_state.table_coords = table_coords

    # Step 4: Crop and save table images
    cropped_dir = f"cropped_{os.path.basename(pdf_path).split('.')[0]}"
    os.makedirs(cropped_dir, exist_ok=True)

    for page, coords in table_coords.items():
        for bbox in coords:
            page_path = os.path.join(image_dir, page)
            output_img_name = f"{page}_{bbox}.png"
            output_path = os.path.join(cropped_dir, output_img_name)
            extractor.crop_table_region(page_path, bbox, output_path)

    st.session_state.cropped_dir = cropped_dir
    st.success("Cropped table images saved.")

# Step 5: Image selection and processing
if st.session_state.cropped_dir:
    st.subheader("📸 Select a Cropped Table Image")
    cropped_images = os.listdir(st.session_state.cropped_dir)
    selected_image = st.selectbox("Choose an image", cropped_images)

    if selected_image:
        full_image_path = os.path.join(st.session_state.cropped_dir, selected_image)
        st.image(full_image_path, caption="Selected Table", use_column_width=True)

        # Step 6: OCR and Analysis
        if st.button("🔍 Extract & Analyze Table"):
            with st.spinner("Running OCR and OpenAI analysis..."):
                text = st.session_state.extractor.extract_text_from_image(
                    st.session_state.cropped_dir, selected_image
                )
                # st.text_area("📝 OCR Text", text, height=200)

                raw_response = st.session_state.extractor.extract_numerics(
                    st.session_state.cropped_dir, selected_image, text
                )
                cleaned_json = st.session_state.extractor.remove_backticks(raw_response)
                cleaned_json = json.loads(cleaned_json)
                try:
                    
                    results = extractor.verify_totals(cleaned_json)
                    for res in results:
                        st.markdown(res)
                   
                except json.JSONDecodeError:
                    st.error("❌ Failed to parse JSON from model response.")
                    st.text(cleaned_json)
