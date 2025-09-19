import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic similarity will not work.")
from typing import Dict, List, Tuple, Optional, Any
import string

class SimilarityCalculator:
    """
    Handles calculation of similarity between correct answers and student answers.
    """
    
    def __init__(self, method: str = "cosine"):
        """
        Initialize similarity calculator.
        
        Args:
            method: Similarity calculation method ("cosine", "semantic", "fuzzy")
        """
        self.method = method
        self.vectorizer = TfidfVectorizer(
            stop_words=None,  # Don't remove stop words for OCR text
            lowercase=True,
            ngram_range=(1, 2),
            max_features=1000,
            min_df=1,  # Include words that appear in at least 1 document
            max_df=0.95  # Don't exclude common words
        )
        
        # Initialize semantic model if needed
        if method == "semantic":
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    raise ImportError(f"Could not load semantic model: {e}")
            else:
                raise ImportError("sentence-transformers not available for semantic similarity")
    
    def calculate_similarity(self, correct_answer: str, student_answer: str) -> float:
        """
        Calculate similarity between correct and student answers.
        
        Args:
            correct_answer: The correct/reference answer
            student_answer: The student's answer
            
        Returns:
            Similarity score between 0 and 1
        """
        if not correct_answer.strip() or not student_answer.strip():
            return 0.0
        
        # Preprocess both answers
        correct_cleaned = self.preprocess_text(correct_answer)
        student_cleaned = self.preprocess_text(student_answer)
        
        # Debug: Print preprocessing results
        print(f"\n🔍 SIMILARITY DEBUG:")
        print(f"Original correct: {correct_answer[:100]}...")
        print(f"Cleaned correct: {correct_cleaned[:100]}...")
        print(f"Original student: {student_answer[:100]}...")
        print(f"Cleaned student: {student_cleaned[:100]}...")
        
        if not correct_cleaned or not student_cleaned:
            print("Warning: One or both answers became empty after preprocessing")
            return 0.0
        
        similarity_score = 0.0
        
        if self.method == "cosine":
            similarity_score = self._cosine_similarity(correct_cleaned, student_cleaned)
        elif self.method == "semantic":
            similarity_score = self._semantic_similarity(correct_cleaned, student_cleaned)
        elif self.method == "fuzzy":
            similarity_score = self._fuzzy_similarity(correct_cleaned, student_cleaned)
        else:
            raise ValueError(f"Unknown similarity method: {self.method}")
        
        print(f"📊 Final similarity score ({self.method}): {similarity_score:.3f}")
        return similarity_score
    
    def _cosine_similarity(self, correct_answer: str, student_answer: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors with robust handling."""
        try:
            # Fit vectorizer on both texts
            corpus = [correct_answer, student_answer]
            
            print(f"  🔍 TF-IDF Debug: Corpus = {corpus}")
            
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Check if vectors are empty
            if tfidf_matrix.shape[1] == 0:
                print("  Warning: TF-IDF produced empty vectors (no vocabulary overlap)")
                return 0.0
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            score = float(similarity_matrix[0][0])
            
            print(f"  📊 TF-IDF matrix shape: {tfidf_matrix.shape}")
            print(f"  📊 Vocabulary size: {len(self.vectorizer.get_feature_names_out()) if hasattr(self.vectorizer, 'get_feature_names_out') else 'unknown'}")
            
            return score
            
        except Exception as e:
            print(f"Error in cosine similarity calculation: {e}")
            return 0.0
    
    def _semantic_similarity(self, correct_answer: str, student_answer: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        try:
            # Encode sentences
            embeddings = self.semantic_model.encode([correct_answer, student_answer])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
            return float(similarity[0][0])
        except Exception as e:
            print(f"Error in semantic similarity calculation: {e}")
            return self._cosine_similarity(correct_answer, student_answer)
    
    def _fuzzy_similarity(self, correct_answer: str, student_answer: str) -> float:
        """Calculate enhanced fuzzy similarity using multiple approaches."""
        try:
            # Method 1: Jaccard similarity (word overlap)
            correct_words = set(correct_answer.lower().split())
            student_words = set(student_answer.lower().split())
            
            intersection = len(correct_words.intersection(student_words))
            union = len(correct_words.union(student_words))
            
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # Method 2: Word coverage (how much of correct answer is covered)
            coverage_score = intersection / len(correct_words) if len(correct_words) > 0 else 0.0
            
            # Method 3: Length similarity bonus
            len_correct = len(correct_answer.split())
            len_student = len(student_answer.split())
            length_similarity = min(len_student, len_correct) / max(len_student, len_correct) if max(len_student, len_correct) > 0 else 0.0
            
            # Combine scores with weights
            final_score = (jaccard_score * 0.6) + (coverage_score * 0.3) + (length_similarity * 0.1)
            
            print(f"  📊 Fuzzy breakdown: Jaccard={jaccard_score:.3f}, Coverage={coverage_score:.3f}, Length={length_similarity:.3f}")
            print(f"  🔄 Word overlap: {intersection}/{union} words")
            
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            print(f"Error in fuzzy similarity calculation: {e}")
            return 0.0
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for similarity calculation - more conservative approach.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove bullet points and formatting characters but keep content
        text = re.sub(r'^[\u2022\u25e6\u25aa\u25ab\u25cf\u25cb\u00b7]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'[\u2022\u25e6\u25aa\u25ab\u25cf\u25cb\u00b7]', ' ', text)
        
        # Remove only truly unnecessary punctuation, keep important ones
        # Keep: periods, commas, colons, semicolons, hyphens
        text = re.sub(r'["\'’`~@#$%^&*()\[\]{}|\\/<>_+=]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Don't remove short words - they might be important (like 'a', 'i', 'is', 'of')
        # This was causing issues with vocabulary overlap
        
        return text

class GradingEngine:
    """
    Main engine for grading student answers.
    """
    
    def __init__(self, similarity_method: str = "cosine"):
        """
        Initialize grading engine.
        
        Args:
            similarity_method: Method for similarity calculation
        """
        self.similarity_calculator = SimilarityCalculator(similarity_method)
        self.grading_thresholds = {
            'excellent': 0.6,    # 60%+ similarity (even more lenient for OCR text)
            'good': 0.45,        # 45-59% similarity
            'satisfactory': 0.3, # 30-44% similarity
            'poor': 0.15,        # 15-29% similarity
            'very_poor': 0.0     # <15% similarity
        }
    
    def grade_answer(self, correct_answer: str, student_answer: str, max_marks: int) -> Dict[str, Any]:
        """
        Grade a single answer.
        
        Args:
            correct_answer: The correct/reference answer
            student_answer: The student's answer
            max_marks: Maximum marks for this question
            
        Returns:
            Dictionary containing grading results
        """
        if not student_answer.strip():
            return {
                'similarity_score': 0.0,
                'marks_awarded': 0,
                'max_marks': max_marks,
                'percentage': 0.0,
                'grade_category': 'no_answer',
                'confidence': 1.0
            }
        
        # Calculate similarity
        similarity_score = self.similarity_calculator.calculate_similarity(
            correct_answer, student_answer
        )
        
        # Calculate marks
        marks_awarded = self._calculate_marks(similarity_score, max_marks)
        
        # Determine grade category
        grade_category = self._get_grade_category(similarity_score)
        
        # Debug: Print grading decision
        print(f"🎯 GRADING RESULT:")
        print(f"Similarity: {similarity_score:.3f} -> Category: {grade_category} -> Marks: {marks_awarded}/{max_marks}")
        
        # Calculate confidence (simple heuristic based on answer length and similarity)
        confidence = self._calculate_confidence(correct_answer, student_answer, similarity_score)
        
        return {
            'similarity_score': similarity_score,
            'marks_awarded': marks_awarded,
            'max_marks': max_marks,
            'percentage': (marks_awarded / max_marks) * 100 if max_marks > 0 else 0,
            'grade_category': grade_category,
            'confidence': confidence
        }
    
    def _calculate_marks(self, similarity_score: float, max_marks: int) -> int:
        """Calculate marks based on similarity score - more generous for OCR text."""
        if similarity_score >= self.grading_thresholds['excellent']:
            return max_marks  # 100% for excellent
        elif similarity_score >= self.grading_thresholds['good']:
            return int(max_marks * 0.9)   # 90% of max marks (more generous)
        elif similarity_score >= self.grading_thresholds['satisfactory']:
            return int(max_marks * 0.75)  # 75% of max marks (more generous)
        elif similarity_score >= self.grading_thresholds['poor']:
            return int(max_marks * 0.6)   # 60% of max marks (more generous)
        else:
            return int(max_marks * 0.3)   # 30% of max marks (some effort)
    
    def _get_grade_category(self, similarity_score: float) -> str:
        """Get grade category based on similarity score."""
        if similarity_score >= self.grading_thresholds['excellent']:
            return 'excellent'
        elif similarity_score >= self.grading_thresholds['good']:
            return 'good'
        elif similarity_score >= self.grading_thresholds['satisfactory']:
            return 'satisfactory'
        elif similarity_score >= self.grading_thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'
    
    def _calculate_confidence(self, correct_answer: str, student_answer: str, 
                            similarity_score: float) -> float:
        """Calculate confidence in the grading decision."""
        # Length ratio (answers should be somewhat similar in length)
        correct_len = len(correct_answer.split())
        student_len = len(student_answer.split())
        
        if correct_len == 0:
            length_ratio = 0.0
        else:
            length_ratio = min(student_len / correct_len, correct_len / student_len)
        
        # Combine similarity and length ratio for confidence
        confidence = (similarity_score + length_ratio) / 2
        
        # Boost confidence for very high or very low similarity scores
        if similarity_score > 0.9 or similarity_score < 0.2:
            confidence = min(confidence * 1.2, 1.0)
        
        return confidence
    
    def grade_all_answers(self, answer_key: Dict[int, Dict[str, Any]], 
                         student_answers: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
        """
        Grade all student answers against the answer key.
        
        Args:
            answer_key: Dictionary of correct answers and marks
            student_answers: Dictionary of student answers
            
        Returns:
            Dictionary of grading results for each question
        """
        results = {}
        
        for q_num, answer_data in answer_key.items():
            correct_answer = answer_data['answer']
            max_marks = answer_data['marks']
            student_answer = student_answers.get(q_num, "")
            
            results[q_num] = self.grade_answer(correct_answer, student_answer, max_marks)
            results[q_num]['correct_answer'] = correct_answer
            results[q_num]['student_answer'] = student_answer
        
        # Add summary statistics
        total_marks = sum(result['marks_awarded'] for result in results.values())
        total_max_marks = sum(result['max_marks'] for result in results.values())
        
        results['summary'] = {
            'total_marks': total_marks,
            'total_max_marks': total_max_marks,
            'percentage': (total_marks / total_max_marks * 100) if total_max_marks > 0 else 0,
            'questions_attempted': len([q for q, r in results.items() 
                                      if q != 'summary' and r['student_answer'].strip()]),
            'total_questions': len(answer_key)
        }
        
        return results

def get_similarity_method_info() -> Dict[str, str]:
    """Get information about available similarity methods."""
    return {
        'fuzzy': 'Enhanced fuzzy similarity - best for concept matching and OCR text (RECOMMENDED)',
        'cosine': 'TF-IDF based cosine similarity - good for exact keyword matching',
        'semantic': 'Semantic similarity using sentence transformers - understands meaning'
    }

class GeminiOnlyGradingEngine:
    """
    Gemini-only grading engine that uses exclusively Gemini AI for intelligent grading.
    No fallbacks - requires Gemini API key to function.
    """
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize Gemini-only grading engine.
        
        Args:
            gemini_api_key: API key for Gemini intelligent grading (required)
        """
        if not gemini_api_key:
            raise ValueError("Gemini API key is required for Gemini-only grading")
        
        try:
            from gemini_processor import GeminiGrader
            self.gemini_grader = GeminiGrader(gemini_api_key)
            print("Gemini-only grading engine initialized successfully")
        except ImportError as e:
            raise ImportError(f"Failed to import Gemini processor: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini grader: {e}")
    
    def grade_answer(self, question_num: int, correct_answer: str, student_answer: str, max_marks: int) -> Dict[str, Any]:
        """
        Grade a single answer using Gemini AI exclusively.
        """
        if not student_answer.strip():
            print(f"Q{question_num}: No student answer provided")
            return {
                'marks_awarded': 0,
                'max_marks': max_marks,
                'percentage': 0.0,
                'similarity_score': 0.0,
                'justification': 'No answer provided by student',
                'suggestions': 'Please provide an answer to receive marks',
                'confidence': 'high',
                'correct_answer': correct_answer,
                'student_answer': student_answer,
                'grading_method': 'gemini'
            }
        
        try:
            print(f"Using Gemini AI for intelligent grading of Q{question_num}...")
            result = self.gemini_grader.grade_answer(question_num, correct_answer, student_answer, max_marks)
            
            # Ensure all required fields are present
            result['correct_answer'] = correct_answer
            result['student_answer'] = student_answer
            result['confidence'] = result.get('confidence', 'high')
            result['grading_method'] = 'gemini'
            
            print(f"Gemini grading successful: {result['marks_awarded']}/{max_marks}")
            return result
            
        except Exception as e:
            print(f"Gemini grading failed for Q{question_num}: {e}")
            print("🚫 No fallback available - Gemini-only mode")
            
            # Return error result instead of fallback
            return {
                'marks_awarded': 0,
                'max_marks': max_marks,
                'percentage': 0.0,
                'similarity_score': 0.0,
                'justification': f'Gemini grading failed: {str(e)}',
                'suggestions': 'Please check your Gemini API key and try again',
                'confidence': 'low',
                'correct_answer': correct_answer,
                'student_answer': student_answer,
                'grading_method': 'gemini_error'
            }
    
    def grade_all_answers(self, answer_key: Dict[int, Dict[str, Any]], 
                         student_answers: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
        """
        Grade all answers using Gemini AI exclusively.
        """
        print("🚀 Starting Gemini-only grading process...")
        results = {}
        
        gemini_success = 0
        gemini_errors = 0
        
        for q_num, answer_data in answer_key.items():
            correct_answer = answer_data['answer']
            max_marks = answer_data['marks']
            student_answer = student_answers.get(q_num, "")
            
            result = self.grade_answer(q_num, correct_answer, student_answer, max_marks)
            results[q_num] = result
            
            # Track success rates
            if result.get('grading_method') == 'gemini':
                gemini_success += 1
            else:
                gemini_errors += 1
        
        # Add comprehensive summary
        total_marks = sum(result['marks_awarded'] for result in results.values())
        total_max_marks = sum(result['max_marks'] for result in results.values())
        
        results['summary'] = {
            'total_marks': total_marks,
            'total_max_marks': total_max_marks,
            'percentage': (total_marks / total_max_marks * 100) if total_max_marks > 0 else 0,
            'questions_attempted': len([q for q, r in results.items() 
                                      if q != 'summary' and r['student_answer'].strip()]),
            'total_questions': len(answer_key),
            'gemini_success': gemini_success,
            'gemini_errors': gemini_errors,
            'grading_method': 'gemini_only'
        }
        
        # Report grading results
        total_qs = len(answer_key)
        print(f"Gemini-only grading complete!")
        print(f"Gemini successful: {gemini_success}/{total_qs} questions")
        if gemini_errors > 0:
            print(f"Gemini errors: {gemini_errors}/{total_qs} questions")
        print(f"📊 Total score: {total_marks}/{total_max_marks} ({results['summary']['percentage']:.1f}%)")
        
        return results