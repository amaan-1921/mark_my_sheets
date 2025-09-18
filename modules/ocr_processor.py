import re
import easyocr
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class OCRProcessor:
    """
    Handles OCR text extraction and question parsing from handwritten answer sheets.
    """
    
    def __init__(self, lang: str = "en", optimize_for_handwriting: bool = False):
        """Initialize EasyOCR with specified language and handwriting optimization."""
        # Initialize EasyOCR reader
        # gpu=False for CPU usage, set to True if you have CUDA GPU
        self.ocr = easyocr.Reader([lang], gpu=False)
        self.optimize_for_handwriting = optimize_for_handwriting
    
    def extract_text_with_confidence(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract text with confidence scores for quality assessment.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing text, confidence, and bounding box info
        """
        extracted_data = self.extract_text_from_image(image_path)
        
        results = []
        for text, confidence, box in extracted_data:
            results.append({
                'text': text,
                'confidence': confidence,
                'bbox': box
            })
        
        return results
    
    def extract_text_from_image(self, image_path: str, preprocess: bool = True) -> List[Tuple[str, float, List[List[int]]]]:
        """
        Extract text from image using OCR with optional preprocessing for handwritten text.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to apply image preprocessing for better OCR
            
        Returns:
            List of tuples (text, confidence, bounding_box)
        """
        try:
            # Apply preprocessing if requested (especially for handwritten text)
            if preprocess or self.optimize_for_handwriting:
                processed_image = self._preprocess_image(image_path)
            else:
                processed_image = image_path
            
            # EasyOCR parameters optimized for handwritten text
            if self.optimize_for_handwriting:
                # Lower width_ths for better character separation in handwriting
                # Lower height_ths for smaller text
                # Adjust other parameters for handwritten content
                results = self.ocr.readtext(
                    processed_image,
                    width_ths=0.4,  # Lower threshold for handwriting
                    height_ths=0.4,  # Lower threshold for handwriting
                    paragraph=False,  # Don't group into paragraphs
                    detail=1  # Return detailed results
                )
            else:
                # Standard parameters for printed text
                results = self.ocr.readtext(processed_image)
            
            if not results:
                return []
            
            extracted_data = []
            for detection in results:
                if isinstance(detection, (list, tuple)) and len(detection) >= 3:
                    # EasyOCR returns [bbox, text, confidence]
                    try:
                        box = detection[0]
                        text = detection[1] 
                        confidence = detection[2]
                        
                        # Apply stricter confidence threshold for handwritten text
                        min_confidence = 0.3 if self.optimize_for_handwriting else 0.5
                        if isinstance(confidence, (int, float)) and confidence >= min_confidence:
                            extracted_data.append((str(text), float(confidence), box))
                    except (IndexError, TypeError, ValueError):
                        continue
            
            return extracted_data
        except Exception as e:
            print(f"Error in OCR extraction: {str(e)}")
            return []
    
    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image for better OCR results on handwritten text.
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Path to the preprocessed image
        """
        try:
            import cv2
            import tempfile
            import os
            
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding for better text extraction
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Save preprocessed image to temporary file
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            temp_path = os.path.join(temp_dir, f"{name}_processed{ext}")
            
            cv2.imwrite(temp_path, cleaned)
            return temp_path
            
        except Exception as e:
            print(f"Warning: Image preprocessing failed: {e}")
            return image_path
    
    def parse_questions_from_text(self, extracted_data: List[Tuple[str, float, List[List[int]]]]) -> Dict[int, str]:
        """
        Parse questions and answers from extracted text data.
        
        Args:
            extracted_data: List of (text, confidence, bounding_box) tuples
            
        Returns:
            Dictionary mapping question numbers to answer text
        """
        questions = {}
        current_question = None
        current_answer = []
        
        # Sort by vertical position (top to bottom)
        # EasyOCR box format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        try:
            sorted_data = sorted(extracted_data, key=lambda x: min(point[1] for point in x[2]) if isinstance(x[2], list) and len(x[2]) > 0 else 0)
        except (IndexError, TypeError):
            # Fallback: just use the data as is
            sorted_data = extracted_data
        
        for text, confidence, box in sorted_data:
            if confidence < 0.5:  # Skip low confidence text
                continue
            
            # Check if this line contains a question number
            question_match = self._find_question_number(text)
            
            if question_match:
                # Save previous question if exists
                if current_question is not None and current_answer:
                    questions[current_question] = ' '.join(current_answer).strip()
                
                # Start new question
                current_question = question_match
                current_answer = []
                
                # Extract any answer text after the question number
                remaining_text = re.sub(r'^\d+\.?\s*', '', text).strip()
                if remaining_text:
                    current_answer.append(remaining_text)
            else:
                # Add to current answer if we have a current question
                if current_question is not None:
                    current_answer.append(text.strip())
        
        # Don't forget the last question
        if current_question is not None and current_answer:
            questions[current_question] = ' '.join(current_answer).strip()
        
        return questions
    
    def _find_question_number(self, text: str) -> Optional[int]:
        """
        Find question number in text.
        
        Args:
            text: Text to search for question number
            
        Returns:
            Question number if found, None otherwise
        """
        # Look for patterns like "1.", "1)", "Q1", "Question 1", etc.
        patterns = [
            r'^(\d+)\.', # 1.
            r'^(\d+)\)', # 1)
            r'^Q\.?(\d+)', # Q1 or Q.1
            r'^Question\s+(\d+)', # Question 1
            r'^\((\d+)\)', # (1)
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip(), re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_questions_from_image(self, image_path: str) -> Dict[int, str]:
        """
        Complete pipeline: extract text and parse questions from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping question numbers to answer text
        """
        extracted_data = self.extract_text_from_image(image_path)
        return self.parse_questions_from_text(extracted_data)
    
    def visualize_extraction(self, image_path: str, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize OCR extraction results on the image.
        
        Args:
            image_path: Path to the image file
            save_path: Optional path to save the visualization
            
        Returns:
            Image with bounding boxes drawn
        """
        extracted_data = self.extract_text_from_image(image_path)
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        for text, confidence, box in extracted_data:
            if confidence > 0.5:  # Only show high confidence detections
                # Convert box to integer points
                points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                
                # Draw bounding box
                cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Add text label
                x, y = int(box[0][0]), int(box[0][1])
                cv2.putText(image, f"{text[:20]}...", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        if save_path:
            cv2.imwrite(save_path, image)
        
        return image


# Utility functions for question processing
def clean_answer_text(text: str) -> str:
    """
    Clean and normalize answer text.
    
    Args:
        text: Raw answer text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common OCR artifacts
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_questions(questions: Dict[int, str], min_length: int = 5) -> Dict[int, str]:
    """
    Validate and filter questions based on minimum length.
    
    Args:
        questions: Dictionary of question number to answer text
        min_length: Minimum length for valid answers
        
    Returns:
        Filtered dictionary of valid questions
    """
    valid_questions = {}
    
    for q_num, answer in questions.items():
        cleaned_answer = clean_answer_text(answer)
        if len(cleaned_answer) >= min_length:
            valid_questions[q_num] = cleaned_answer
    
    return valid_questions