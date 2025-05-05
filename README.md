steps to run the code


# you need to add key in file Openai_api.py
self.client = OpenAI(api_key="key")

# make environment
python -m venv test_env

source test_env/bin/activate

# installations
pip install -r requirements.txt

# call app
streamlit run app.py

what to do ?
1. uplaoad any pdf
2. select an image that you need to be analysed
3. see the results

if you find any issue, related to pytesseract 
you need make changes in extract_info.py
As i am currently using MacOS, currently command is written "pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'"

for linux "pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'"
for windows "pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'"





