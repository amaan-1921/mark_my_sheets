# MarkMySheets 📝

An AI-powered automated grading system for handwritten answer sheets.

## Features
- OCR text extraction from handwritten answers
- Cosine similarity-based answer comparison
- LLM-powered grading justifications
- Streamlit web interface

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Usage
1. Upload answer key (PDF/Word document)
2. Upload student answer sheet (image)
3. Set question weights
4. Get automated grading with justifications

## Project Structure
```
MarkMySheets/
├── app.py                 # Main Streamlit application
├── modules/
│   ├── ocr_processor.py   # OCR and text extraction
│   ├── answer_parser.py   # Question parsing and structuring
│   ├── similarity.py     # Answer similarity calculation
│   └── llm_grader.py     # LLM integration for justifications
├── utils/
│   └── file_handlers.py  # File upload and processing utilities
└── data/                 # Sample data and temporary files
```