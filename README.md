# MarkMySheets 📝

An AI-powered automated grading system for handwritten answer sheets using **Gemini 2.5 Flash exclusively**.

## Features

## Prerequisites
- Google AI Studio API key (free tier available)
- Python 3.8+

## Setup
1. Get your free Gemini API key from [Google AI Studio](https://aistudio.google.com/)

2. Set your API key as an environment variable:
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage
1. Ensure your Gemini API key is configured
2. Upload answer key (PDF/Word document) - Gemini will intelligently extract questions and answers
3. Upload student answer sheet (image) - Gemini OCR will process handwritten text with advanced preprocessing
4. Set question weights if needed
5. Get comprehensive AI-powered grading with detailed justifications

## Project Structure
```
MarkMySheets/
├── app.py                 # Main Streamlit application
├── modules/
│   ├── answer_parser.py   # Question parsing and structuring
│   ├── gemini_processor.py # Gemini AI for OCR and text extraction
│   ├── image_preprocessor.py # Advanced image preprocessing for Image/Video Analytics
│   ├── similarity.py     # Gemini-only grading engine
│   └── llm_grader.py     # LLM integration for justifications
├── utils/
│   └── file_handlers.py  # File upload and processing utilities
└── data/                 # Sample data and temporary files
```