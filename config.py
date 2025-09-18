# Configuration file for MarkMySheets

# Default settings
DEFAULT_SIMILARITY_METHOD = "cosine"
DEFAULT_LLM_PROVIDER = "mock"
MAX_FILE_SIZE_MB = 10

# Supported file types
SUPPORTED_ANSWER_KEY_TYPES = ["pdf", "word"]
SUPPORTED_IMAGE_TYPES = ["image"]

# Grading thresholds (adjusted for OCR-extracted text)
GRADING_THRESHOLDS = {
    'excellent': 0.7,    # 70%+ similarity (realistic for OCR text)
    'good': 0.55,        # 55-69% similarity
    'satisfactory': 0.4, # 40-54% similarity
    'poor': 0.25,        # 25-39% similarity
    'very_poor': 0.0     # <25% similarity
}

# Mark calculation percentages
MARK_PERCENTAGES = {
    'excellent': 1.0,
    'good': 0.85,
    'satisfactory': 0.7,
    'poor': 0.5,
    'very_poor': 0.2
}

# LLM API endpoints
LLM_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models"
}

# OCR settings
OCR_CONFIDENCE_THRESHOLD = 0.5
MIN_ANSWER_LENGTH = 5

# UI settings
STREAMLIT_CONFIG = {
    "page_title": "MarkMySheets - AI Grading Assistant",
    "page_icon": "📝",
    "layout": "wide"
}