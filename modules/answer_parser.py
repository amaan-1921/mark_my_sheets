import re
import json
import base64
from typing import Dict, Any, Optional
from docx import Document
import PyPDF2
import io

# Gemini import with fallback
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class AnswerKeyParser:
    """
    Gemini-powered answer key parser for intelligent text extraction.
    Extracts questions, answers, and marks from PDF/Word documents.
    """
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize the parser with Gemini AI.
        
        Args:
            gemini_api_key: Gemini API key for AI processing
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        
        if self.gemini_api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("✅ Gemini AI initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini: {e}")
                self.gemini_model = None
        else:
            print("⚠️ Gemini not available - API key missing or module not installed")
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF using Gemini AI."""
        if not self.gemini_model:
            raise ValueError("Gemini AI not initialized. Please provide a valid API key.")
        
        try:
            # Encode PDF for Gemini
            pdf_base64 = base64.b64encode(file_content).decode('utf-8')
            
            prompt = """
Analyze this PDF answer key and extract ALL questions and answers.

Return ONLY a JSON object in this exact format:
{
  "questions": {
    "1": {
      "question": "Complete question text (without Q1. prefix)",
      "answer": "Complete answer text (without Answer: prefix)",
      "marks": 10
    },
    "2": {
      "question": "Complete question text (without Q2. prefix)", 
      "answer": "Complete answer text (without Answer: prefix)",
      "marks": 10
    }
  }
}

IMPORTANT:
1. Extract COMPLETE answer text - don't summarize or truncate
2. Remove "Q1.", "Question 1:", "Answer:" prefixes from the extracted text
3. If marks are specified ("[10 marks]", "(5 points)"), use that number, otherwise default to 10
4. Preserve bullet points (●, ○) and formatting in answers
5. Return ONLY the JSON - no extra text
"""
            
            file_part = {
                "mime_type": "application/pdf",
                "data": pdf_base64
            }
            
            print("🤖 Using Gemini AI to extract answer key...")
            response = self.gemini_model.generate_content([prompt, file_part])
            
            if response and response.text:
                return response.text
            else:
                raise ValueError("API failed: No response received from Gemini AI")
                
        except Exception as e:
            raise ValueError(f"API failed: Gemini processing error - {str(e)}")
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from Word document and process with Gemini."""
        if not self.gemini_model:
            raise ValueError("Gemini AI not initialized. Please provide a valid API key.")
        
        try:
            # Extract text from Word document
            doc = Document(io.BytesIO(file_content))
            full_text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            
            # Process with Gemini
            prompt = f"""
Analyze this answer key text and extract ALL questions and answers.

Text:
{full_text}

Return ONLY a JSON object in this exact format:
{{
  "questions": {{
    "1": {{
      "question": "Complete question text (without Q1. prefix)",
      "answer": "Complete answer text (without Answer: prefix)",
      "marks": 10
    }}
  }}
}}

IMPORTANT:
1. Extract COMPLETE answer text - don't summarize or truncate
2. Remove "Q1.", "Question 1:", "Answer:" prefixes
3. If marks specified, use that number, otherwise default to 10
4. Preserve bullet points and formatting
5. Return ONLY the JSON
"""
            
            print("🤖 Using Gemini AI to process Word document...")
            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text:
                return response.text
            else:
                raise ValueError("API failed: No response received from Gemini AI")
                
        except Exception as e:
            raise ValueError(f"API failed: Word document processing error - {str(e)}")
    
    def parse_answer_key(self, gemini_response: str) -> Dict[int, Dict[str, Any]]:
        """
        Parse Gemini's JSON response into the expected format.
        Returns: {question_number: {'question': str, 'answer': str, 'marks': int}}
        """
        try:
            # Clean response text
            response_text = gemini_response.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            data = json.loads(response_text)
            questions_data = data.get('questions', {})
            
            # Convert to expected format with question text included
            result = {}
            for q_num_str, q_data in questions_data.items():
                q_num = int(q_num_str)
                result[q_num] = {
                    'question': q_data.get('question', '').strip(),
                    'answer': q_data.get('answer', '').strip(),
                    'marks': q_data.get('marks', 10)
                }
            
            print(f"✅ Successfully extracted {len(result)} questions")
            for q_num, data in result.items():
                print(f"   Q{q_num}: Question ({len(data['question'])} chars), Answer ({len(data['answer'])} chars), {data['marks']} marks")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {response_text[:500]}...")
            raise ValueError(f"API failed: Could not parse Gemini response - {e}")
        except Exception as e:
            raise ValueError(f"API failed: Error processing Gemini response - {e}")

class QuestionMatcher:
    """
    Handles matching questions between answer key and student answers.
    """
    
    def match_questions(self, answer_key: Dict[int, Dict[str, Any]], 
                       student_answers: Dict[int, str]) -> list:
        """
        Match questions between answer key and student answers.
        """
        matches = []
        
        for q_num in sorted(answer_key.keys()):
            correct_answer = answer_key[q_num]['answer']
            marks = answer_key[q_num]['marks']
            student_answer = student_answers.get(q_num, "")
            
            matches.append((q_num, correct_answer, student_answer, marks))
        
        return matches