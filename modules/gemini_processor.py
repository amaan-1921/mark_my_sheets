"""
Gemini 2.5 Flash integration for handwritten OCR and LLM grading
Uses Google AI Studio free API for both text extraction and grading justification

Features:
- Advanced image preprocessing for image/video analytics projects
- Multi-stage noise reduction and contrast enhancement
- Adaptive thresholding for varying lighting conditions
- Morphological operations optimized for handwritten text
- Edge-preserving filters and sharpening techniques
"""
import google.generativeai as genai
import base64
import re
import json
import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageEnhance, ImageFilter
import io

class GeminiHandwritingOCR:
    """
    Gemini 2.5 Flash powered handwriting OCR with advanced image preprocessing.
    
    Features for Image/Video Analytics:
    - Bilateral filtering for noise reduction while preserving edges
    - Histogram equalization for contrast enhancement
    - Dual adaptive thresholding (Gaussian + Mean) for robustness
    - Morphological operations optimized for text connectivity
    - PIL-based fine-tuning for sharpness and contrast
    
    Perfect for:
    - Handwritten answer sheets from photos/scans
    - Low-quality video frames with text
    - Images with varying lighting conditions
    - Noisy or blurred text documents
    """
    
    def __init__(self, api_key: str, enable_preprocessing: bool = True):
        """Initialize Gemini with API key from Google AI Studio"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.enable_preprocessing = enable_preprocessing
    
    def extract_handwritten_questions(self, image_path: str) -> Dict[int, str]:
        """
        Extract handwritten questions and answers from image using Gemini
        
        Args:
            image_path: Path to the handwritten answer sheet image
            
        Returns:
            Dictionary mapping question numbers to student answers
        """
        try:
            # Load and prepare image
            image_path_processed = image_path
            
            # Apply advanced preprocessing for image/video analytics if enabled
            if self.enable_preprocessing:
                image_path_processed = self._preprocess_image_for_analytics(image_path)
                print(f"🖼️ Applied advanced image preprocessing for optimal OCR")
            
            image = Image.open(image_path_processed)
            
            # Create prompt optimized for handwritten answer sheet extraction
            prompt = """
You are an expert OCR system specialized in reading handwritten answer sheets. 

Please analyze this handwritten answer sheet and extract the text with high accuracy. 

IMPORTANT INSTRUCTIONS:
1. This is a student's handwritten answer sheet with numbered questions
2. Look for question numbers like: 1., 2., 3., Q1, Q2, Question 1, etc.
3. Extract the complete answer text for each question
4. Maintain the original writing as much as possible
5. If handwriting is unclear, make your best interpretation
6. Format your response as JSON with this structure:

{
  "questions": {
    "1": "student's answer for question 1",
    "2": "student's answer for question 2",
    "3": "student's answer for question 3"
  },
  "confidence": "high/medium/low",
  "notes": "any observations about handwriting quality or issues"
}

Focus on accuracy and completeness. Even if some words are unclear, provide your best interpretation.
"""

            # Generate response
            response = self.model.generate_content([prompt, image])
            
            # Parse the response
            return self._parse_gemini_response(response.text)
            
        except Exception as e:
            print(f"Error in Gemini OCR: {str(e)}")
            return {}
    
    def _parse_gemini_response(self, response_text: str) -> Dict[int, str]:
        """Parse Gemini's JSON response to extract questions"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                questions = {}
                if 'questions' in data:
                    for q_num, answer in data['questions'].items():
                        try:
                            questions[int(q_num)] = answer.strip()
                        except (ValueError, AttributeError):
                            continue
                
                # Log confidence and notes
                if 'confidence' in data:
                    print(f"🎯 Gemini OCR Confidence: {data['confidence']}")
                if 'notes' in data:
                    print(f"📝 Notes: {data['notes']}")
                    
                return questions
            else:
                # Fallback: try to parse without JSON structure
                return self._fallback_parse(response_text)
                
        except json.JSONDecodeError:
            print("⚠️ Could not parse JSON response, trying fallback parsing...")
            return self._fallback_parse(response_text)
    
    def _fallback_parse(self, text: str) -> Dict[int, str]:
        """Fallback parsing when JSON parsing fails"""
        questions = {}
        lines = text.split('\n')
        
        current_question = None
        current_answer = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for question patterns
            question_match = re.match(r'^[\"\']?(\d+)[\"\']?\s*[:.]?\s*[\"\']?(.*)', line)
            if question_match:
                # Save previous question
                if current_question and current_answer:
                    questions[current_question] = ' '.join(current_answer).strip()
                
                # Start new question
                current_question = int(question_match.group(1))
                answer_text = question_match.group(2).strip('\'"')
                current_answer = [answer_text] if answer_text else []
            else:
                # Add to current answer
                if current_question:
                    current_answer.append(line.strip('\'"'))
        
        # Don't forget last question
        if current_question and current_answer:
            questions[current_question] = ' '.join(current_answer).strip()
        
        return questions

    def _preprocess_image_for_analytics(self, image_path: str) -> str:
        """
        Advanced image preprocessing for image/video analytics projects.
        Optimized for handwritten text extraction with multiple enhancement techniques.
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Path to the preprocessed image
        """
        try:
            # Create temp directory
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            processed_path = os.path.join(temp_dir, f"{name}_gemini_processed{ext}")
            
            # Load image with OpenCV for advanced processing
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Could not load image {image_path}, using original")
                return image_path
            
            # Step 1: Noise reduction using bilateral filter
            # Preserves edges while reducing noise - crucial for handwritten text
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
            
            # Step 2: Convert to grayscale
            gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Histogram equalization for better contrast
            equalized = cv2.equalizeHist(gray)
            
            # Step 4: Advanced morphological operations for text enhancement
            # Create custom kernel for text enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Opening to remove small noise
            opened = cv2.morphologyEx(equalized, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Step 5: Adaptive thresholding - excellent for varying lighting conditions
            # Two different methods combined for robustness
            thresh1 = cv2.adaptiveThreshold(
                opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
            )
            thresh2 = cv2.adaptiveThreshold(
                opened, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
            )
            
            # Combine both thresholding methods
            combined_thresh = cv2.bitwise_and(thresh1, thresh2)
            
            # Step 6: Advanced morphological closing to connect broken characters
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            closed = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, close_kernel)
            
            # Step 7: Edge-preserving smoothing for final cleanup
            # Use bilateral filter on the binary image
            final = cv2.bilateralFilter(closed, 5, 50, 50)
            
            # Step 8: Optional sharpening for enhanced text clarity
            # Create sharpening kernel
            sharpen_kernel = np.array([[-1, -1, -1],
                                     [-1,  9, -1],
                                     [-1, -1, -1]])
            sharpened = cv2.filter2D(final, -1, sharpen_kernel)
            
            # Step 9: Final contrast enhancement using PIL for fine-tuning
            cv2.imwrite(processed_path, sharpened)
            
            # Load with PIL for additional enhancements
            pil_img = Image.open(processed_path)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(1.2)  # Slight contrast boost
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            final_enhanced = sharpness_enhancer.enhance(1.1)  # Subtle sharpening
            
            # Save final processed image
            final_enhanced.save(processed_path, quality=95, optimize=True)
            
            print(f"✅ Advanced preprocessing complete: {processed_path}")
            return processed_path
            
        except Exception as e:
            print(f"⚠️ Image preprocessing failed: {e}")
            print(f"📝 Using original image: {image_path}")
            return image_path
    
    def _enhance_image_quality(self, image_path: str) -> str:
        """
        Additional image quality enhancement specifically for video analytics.
        Focuses on frame stabilization and quality improvement.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Path to the enhanced image
        """
        try:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            enhanced_path = os.path.join(temp_dir, f"{name}_enhanced{ext}")
            
            # Load with PIL for quality enhancements
            img = Image.open(image_path)
            
            # Apply unsharp mask for better text definition
            unsharp = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=1))
            
            # Brightness and contrast optimization
            brightness_enhancer = ImageEnhance.Brightness(unsharp)
            bright_img = brightness_enhancer.enhance(1.05)  # Slight brightness increase
            
            contrast_enhancer = ImageEnhance.Contrast(bright_img)
            final_img = contrast_enhancer.enhance(1.15)  # Better contrast
            
            # Save enhanced image
            final_img.save(enhanced_path, quality=95)
            return enhanced_path
            
        except Exception as e:
            print(f"⚠️ Image enhancement failed: {e}")
            return image_path


class GeminiGrader:
    """
    Advanced Gemini-powered grading system for intelligent answer evaluation.
    Provides true semantic understanding and educational assessment.
    """
    
    def __init__(self, api_key: str):
        """Initialize Gemini grader with API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.api_key = api_key
        print("🤖 Gemini Grader initialized for intelligent answer evaluation")
    
    def grade_answer(self, question_num: int, correct_answer: str, student_answer: str, 
                    max_marks: int = 10) -> Dict[str, Any]:
        """
        Grade a single answer using Gemini's advanced reasoning capabilities.
        
        Args:
            question_num: Question number
            correct_answer: The reference/correct answer
            student_answer: Student's response
            max_marks: Maximum marks possible for this question
            
        Returns:
            Comprehensive grading results with scores, justification, and feedback
        """
        if not student_answer.strip():
            return {
                'marks_out_of_10': 0,
                'percentage': 0.0,
                'similarity_score': 0.0,
                'confidence': 'high',
                'justification': 'No answer provided by the student.',
                'suggestions': 'Please attempt to answer the question to receive feedback.',
                'grade_category': 'no_answer',
                'grading_method': 'gemini'
            }
        
        try:
            # Create comprehensive grading prompt
            prompt = self._create_grading_prompt(question_num, correct_answer, student_answer, max_marks)
            
            # Get Gemini's assessment
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                result = self._parse_gemini_response(response.text, max_marks)
                result['grading_method'] = 'gemini'
                return result
            else:
                raise Exception("No response from Gemini")
                
        except Exception as e:
            print(f"⚠️ Gemini grading failed: {e}")
            print("🔄 Falling back to similarity-based grading")
            return self._fallback_grading(correct_answer, student_answer, max_marks)
    
    def _create_grading_prompt(self, question_num: int, correct_answer: str, 
                              student_answer: str, max_marks: int) -> str:
        """
        Create a comprehensive prompt for Gemini grading that ensures educational accuracy.
        """
        return f"""
You are an expert teacher grading student answers with deep subject matter expertise. 
Provide a fair, comprehensive evaluation that recognizes conceptual understanding.

QUESTION {question_num} GRADING TASK:

CORRECT/REFERENCE ANSWER:
{correct_answer}

STUDENT'S ANSWER:
{student_answer}

GRADING INSTRUCTIONS:
1. Evaluate conceptual understanding, not just exact word matching
2. Recognize equivalent explanations and terminology
3. Assess completeness, accuracy, and clarity
4. Consider partial credit for partially correct responses
5. Be fair but maintain academic standards

Provide your assessment in this EXACT JSON format:
{{
  "marks_out_of_10": 8,
  "percentage": 80.0,
  "confidence": "high",
  "justification": "Clear explanation of why this grade was assigned, highlighting what the student got right and what could be improved.",
  "suggestions": "Specific advice for improvement and learning.",
  "grade_category": "good"
}}

GRADE CATEGORIES:
- "excellent" (90-100%): Comprehensive, accurate, well-explained
- "good" (75-89%): Good understanding with minor gaps
- "satisfactory" (60-74%): Basic understanding, some important points covered
- "poor" (40-59%): Limited understanding, major gaps
- "very_poor" (0-39%): Minimal understanding or major errors

CONFIDENCE LEVELS: "high", "medium", "low"

IMPORTANT: Return ONLY the JSON response, no additional text.
"""
    
    def _parse_gemini_response(self, response_text: str, max_marks: int) -> Dict[str, Any]:
        """
        Parse Gemini's JSON response and validate the grading results.
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                # Validate and normalize the response
                marks = int(data.get('marks_out_of_10', 0))
                marks = max(0, min(marks, 10))  # Clamp between 0-10
                
                # Scale marks to actual max_marks
                actual_marks = int((marks / 10) * max_marks)
                
                return {
                    'marks_out_of_10': marks,
                    'marks_awarded': actual_marks,
                    'max_marks': max_marks,
                    'percentage': float(data.get('percentage', (marks / 10) * 100)),
                    'similarity_score': marks / 10,  # Use Gemini score as similarity
                    'confidence': data.get('confidence', 'medium'),
                    'justification': data.get('justification', 'Graded by Gemini AI'),
                    'suggestions': data.get('suggestions', 'Keep studying and practicing'),
                    'grade_category': data.get('grade_category', self._get_category_from_percentage((marks / 10) * 100))
                }
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️ Failed to parse Gemini response: {e}")
            print(f"📝 Raw response: {response_text[:200]}...")
            return self._create_fallback_result(max_marks)
    
    def _get_category_from_percentage(self, percentage: float) -> str:
        """Convert percentage to grade category"""
        if percentage >= 90: return 'excellent'
        elif percentage >= 75: return 'good'
        elif percentage >= 60: return 'satisfactory'
        elif percentage >= 40: return 'poor'
        else: return 'very_poor'
    
    def _fallback_grading(self, correct_answer: str, student_answer: str, max_marks: int) -> Dict[str, Any]:
        """
        Fallback to similarity-based grading when Gemini fails.
        """
        try:
            from similarity import SimilarityCalculator
            
            calculator = SimilarityCalculator('fuzzy')  # Use enhanced fuzzy similarity
            similarity_score = calculator.calculate_similarity(correct_answer, student_answer)
            
            # Convert similarity to marks using the improved thresholds
            if similarity_score >= 0.6: marks_out_of_10 = 10
            elif similarity_score >= 0.45: marks_out_of_10 = 9
            elif similarity_score >= 0.3: marks_out_of_10 = 7
            elif similarity_score >= 0.15: marks_out_of_10 = 6
            else: marks_out_of_10 = 3
            
            actual_marks = int((marks_out_of_10 / 10) * max_marks)
            percentage = (marks_out_of_10 / 10) * 100
            
            return {
                'marks_out_of_10': marks_out_of_10,
                'marks_awarded': actual_marks,
                'max_marks': max_marks,
                'percentage': percentage,
                'similarity_score': similarity_score,
                'confidence': 'medium',
                'justification': f'Fallback grading based on text similarity ({similarity_score:.2f}). Gemini grading was unavailable.',
                'suggestions': 'Review the key concepts and try to explain in more detail.',
                'grade_category': self._get_category_from_percentage(percentage),
                'grading_method': 'similarity_fallback'
            }
            
        except Exception as e:
            print(f"⚠️ Fallback grading also failed: {e}")
            return self._create_fallback_result(max_marks)
    
    def _create_fallback_result(self, max_marks: int) -> Dict[str, Any]:
        """Create a basic result when all grading methods fail"""
        return {
            'marks_out_of_10': 5,
            'marks_awarded': max_marks // 2,
            'max_marks': max_marks,
            'percentage': 50.0,
            'similarity_score': 0.5,
            'confidence': 'low',
            'justification': 'Unable to grade automatically. Please review manually.',
            'suggestions': 'Review your answer and consult with instructor.',
            'grade_category': 'satisfactory',
            'grading_method': 'error_fallback'
        }
    """
    Gemini 2.5 Flash powered LLM grader for answer evaluation and justification
    """
    
    def __init__(self, api_key: str):
        """Initialize Gemini LLM grader"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def generate_justification(self, question_num: int, correct_answer: str, 
                             student_answer: str, similarity_score: float, 
                             marks_awarded: float, max_marks: float) -> str:
        """Generate detailed justification for marks awarded"""
        
        prompt = f"""
You are an expert teacher grading student answers. Please provide a detailed justification for the marks awarded.

QUESTION {question_num}:
Correct Answer: {correct_answer}

Student Answer: {student_answer}

GRADING DETAILS:
- Similarity Score: {similarity_score:.2f}
- Marks Awarded: {marks_awarded}/{max_marks}
- Percentage: {(marks_awarded/max_marks)*100:.1f}%

Please provide a clear, constructive justification that explains:
1. What the student got right
2. What was missing or incorrect
3. Why this specific mark was awarded
4. Be encouraging but honest

Keep the justification concise but thorough (2-3 sentences).
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating justification: {str(e)}"
    
    def generate_improvement_suggestions(self, correct_answer: str, student_answer: str) -> str:
        """Generate suggestions for improvement"""
        
        prompt = f"""
As an expert teacher, provide specific, actionable suggestions for how this student can improve their answer.

Correct Answer: {correct_answer}
Student Answer: {student_answer}

Please provide:
1. Specific areas where the student can improve
2. Key concepts they should focus on
3. Study suggestions
4. Be constructive and encouraging

Keep suggestions brief but specific (2-3 points maximum).
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"
    
    def generate_overall_feedback(self, results: Dict[Any, Any]) -> str:
        """Generate overall feedback for the student"""
        
        # Calculate overall statistics
        total_questions = len([k for k in results.keys() if isinstance(k, int)])
        total_marks = sum(results[k]['marks_awarded'] for k in results.keys() if isinstance(k, int))
        total_max_marks = sum(results[k]['max_marks'] for k in results.keys() if isinstance(k, int))
        percentage = (total_marks / total_max_marks * 100) if total_max_marks > 0 else 0
        
        # Identify strong and weak areas
        good_questions = [k for k in results.keys() if isinstance(k, int) and results[k]['percentage'] >= 75]
        weak_questions = [k for k in results.keys() if isinstance(k, int) and results[k]['percentage'] < 50]
        
        prompt = f"""
You are an expert teacher providing overall feedback to a student who completed {total_questions} questions.

PERFORMANCE SUMMARY:
- Total Score: {total_marks}/{total_max_marks} ({percentage:.1f}%)
- Strong Performance: Questions {', '.join(map(str, good_questions)) if good_questions else 'None'}
- Needs Improvement: Questions {', '.join(map(str, weak_questions)) if weak_questions else 'None'}

Please provide encouraging overall feedback that:
1. Acknowledges their efforts and strengths
2. Identifies key areas for improvement
3. Provides motivational guidance
4. Suggests next steps for learning

Keep feedback positive, constructive, and motivating (3-4 sentences).
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Overall performance: {percentage:.1f}%. Keep practicing and focus on understanding key concepts."


class GeminiMarkMySheets:
    """
    Complete Gemini-powered grading system combining OCR and intelligent grading capabilities
    """
    
    def __init__(self, api_key: str, enable_preprocessing: bool = True):
        """Initialize both OCR and intelligent grading components"""
        self.api_key = api_key
        self.ocr = GeminiHandwritingOCR(api_key, enable_preprocessing)
        self.grader = GeminiGrader(api_key)  # Use the new intelligent grader
        print(f"🔧 Gemini system initialized with preprocessing: {enable_preprocessing}")
        print("🎆 Intelligent Gemini grading system ready!")
    
    def process_answer_sheet(self, image_path: str) -> Dict[int, str]:
        """Extract text from handwritten answer sheet with preprocessing confirmation"""
        print("🤖 Processing handwritten answer sheet with Gemini 2.5 Flash...")
        if self.ocr.enable_preprocessing:
            print("🔧 Advanced image preprocessing is ENABLED for optimal OCR results")
        questions = self.ocr.extract_handwritten_questions(image_path)
        print(f"✅ Extracted {len(questions)} questions from handwriting")
        return questions
    
    def grade_with_intelligence(self, question_num: int, correct_answer: str, 
                               student_answer: str, max_marks: int = 10) -> Dict[str, Any]:
        """Grade answer using Gemini's advanced reasoning capabilities"""
        print(f"🤖 Using Gemini intelligent grading for Question {question_num}...")
        result = self.grader.grade_answer(question_num, correct_answer, student_answer, max_marks)
        
        # Add additional fields for compatibility
        result['correct_answer'] = correct_answer
        result['student_answer'] = student_answer
        
        print(f"✅ Gemini grading complete: {result['marks_awarded']}/{max_marks} ({result['percentage']:.1f}%)")
        return result
    
    def batch_grade_answers(self, answer_key: Dict[int, Dict[str, Any]], 
                           student_answers: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
        """Grade all answers using Gemini intelligence with progress tracking"""
        print("🚀 Starting Gemini intelligent batch grading...")
        results = {}
        
        total_questions = len(answer_key)
        
        for i, (q_num, answer_data) in enumerate(answer_key.items(), 1):
            correct_answer = answer_data['answer']
            max_marks = answer_data['marks']
            student_answer = student_answers.get(q_num, "")
            
            print(f"📝 Processing Question {q_num} ({i}/{total_questions})...")
            
            results[q_num] = self.grade_with_intelligence(q_num, correct_answer, student_answer, max_marks)
            
            # Small delay to be respectful to API
            import time
            time.sleep(0.5)
        
        # Calculate summary statistics
        total_marks = sum(result['marks_awarded'] for result in results.values())
        total_max_marks = sum(result['max_marks'] for result in results.values())
        
        results['summary'] = {
            'total_marks': total_marks,
            'total_max_marks': total_max_marks,
            'percentage': (total_marks / total_max_marks * 100) if total_max_marks > 0 else 0,
            'questions_attempted': len([q for q, r in results.items() 
                                      if q != 'summary' and r['student_answer'].strip()]),
            'total_questions': len(answer_key),
            'grading_method': 'gemini_intelligent'
        }
        
        print(f"🎆 Batch grading complete: {total_marks}/{total_max_marks} ({results['summary']['percentage']:.1f}%)")
        return results


# Test function
def test_gemini_ocr():
    """Test Gemini OCR with sample handwritten image"""
    import os
    
    # You'll need to set your Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("💡 Get your free API key from: https://aistudio.google.com/")
        return
    
    gemini = GeminiMarkMySheets(api_key)
    
    # Test with sample1.png
    image_path = "samples/sample1.png"
    if os.path.exists(image_path):
        print("🧪 TESTING GEMINI HANDWRITING OCR")
        print("=" * 50)
        
        questions = gemini.process_answer_sheet(image_path)
        
        if questions:
            print(f"\n📋 EXTRACTED QUESTIONS:")
            print("-" * 30)
            for q_num, answer in questions.items():
                print(f"Question {q_num}: {answer[:100]}...")
        else:
            print("⚠️ No questions extracted")
    else:
        print(f"❌ Test image not found: {image_path}")


if __name__ == "__main__":
    test_gemini_ocr()