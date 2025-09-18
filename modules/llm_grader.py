import requests
import json
from typing import Dict, List, Optional, Any, Union, Union
import os
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GroqProvider(LLMProvider):
    """Groq API provider for LLM interactions."""
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Groq API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class OpenAIProvider(LLMProvider):
    """OpenAI API provider for LLM interactions."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class GeminiProvider(LLMProvider):
    """Google Gemini API provider for LLM interactions."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini API."""
        url = f"{self.base_url}?key={self.api_key}"
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.3
            }
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class LLMGrader:
    """
    Handles LLM-based grading justifications and feedback.
    """
    
    def __init__(self, provider: str = "groq", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM grader.
        
        Args:
            provider: LLM provider ("groq", "openai", "gemini")
            api_key: API key for the provider
            model: Model name to use
        """
        if not api_key:
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key not provided for {provider}. Set {provider.upper()}_API_KEY environment variable.")
        
        if provider.lower() == "groq":
            self.llm: LLMProvider = GroqProvider(api_key, model or "llama3-8b-8192")
        elif provider.lower() == "openai":
            self.llm = OpenAIProvider(api_key, model or "gpt-3.5-turbo")
        elif provider.lower() == "gemini":
            self.llm = GeminiProvider(api_key, model or "gemini-2.0-flash-exp")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.provider_name = provider
    
    def generate_justification(self, question_num: int, correct_answer: str, 
                             student_answer: str, similarity_score: float, 
                             marks_awarded: int, max_marks: int) -> str:
        """
        Generate justification for the assigned marks.
        
        Args:
            question_num: Question number
            correct_answer: The correct/reference answer
            student_answer: The student's answer
            similarity_score: Calculated similarity score
            marks_awarded: Marks awarded to the student
            max_marks: Maximum possible marks
            
        Returns:
            Generated justification text
        """
        if not student_answer.strip():
            return "No answer provided by the student. Full marks deducted."
        
        prompt = self._create_justification_prompt(
            question_num, correct_answer, student_answer, 
            similarity_score, marks_awarded, max_marks
        )
        
        return self.llm.generate_response(prompt)
    
    def generate_improvement_suggestions(self, correct_answer: str, student_answer: str) -> str:
        """
        Generate suggestions for improving the student's answer.
        
        Args:
            correct_answer: The correct/reference answer
            student_answer: The student's answer
            
        Returns:
            Generated improvement suggestions
        """
        if not student_answer.strip():
            return "Please provide an answer to receive feedback."
        
        prompt = self._create_improvement_prompt(correct_answer, student_answer)
        return self.llm.generate_response(prompt)
    
    def generate_overall_feedback(self, grading_results: Dict[Union[int, str], Any]) -> str:
        """
        Generate overall feedback for the student's performance.
        
        Args:
            grading_results: Complete grading results
            
        Returns:
            Generated overall feedback
        """
        summary = grading_results.get('summary', {})
        total_percentage = summary.get('percentage', 0)
        questions_attempted = summary.get('questions_attempted', 0)
        total_questions = summary.get('total_questions', 0)
        
        prompt = self._create_overall_feedback_prompt(
            total_percentage, questions_attempted, total_questions, grading_results
        )
        
        return self.llm.generate_response(prompt)
    
    def _create_justification_prompt(self, question_num: int, correct_answer: str, 
                                   student_answer: str, similarity_score: float, 
                                   marks_awarded: int, max_marks: int) -> str:
        """Create prompt for grading justification."""
        return f"""You are an expert teacher grading student answers. Provide a concise justification for the marks awarded.

Question {question_num}:

Correct Answer: {correct_answer}

Student Answer: {student_answer}

Similarity Score: {similarity_score:.2f}
Marks Awarded: {marks_awarded}/{max_marks}

Please provide a brief justification (2-3 sentences) explaining why these marks were awarded. Focus on:
1. What the student got right
2. What was missing or incorrect
3. The overall quality of the response

Keep the tone constructive and educational."""

    def _create_improvement_prompt(self, correct_answer: str, student_answer: str) -> str:
        """Create prompt for improvement suggestions."""
        return f"""You are a helpful teacher providing feedback to improve student answers.

Correct Answer: {correct_answer}

Student Answer: {student_answer}

Please provide specific, actionable suggestions (2-3 points) on how the student can improve their answer. Focus on:
1. Missing key concepts or information
2. Areas that need more detail or clarity
3. Better ways to structure the response

Be encouraging and specific in your feedback."""

    def _create_overall_feedback_prompt(self, total_percentage: float, 
                                      questions_attempted: int, total_questions: int,
                                      grading_results: Dict[Union[int, str], Any]) -> str:
        """Create prompt for overall feedback."""
        
        # Analyze performance by category
        excellent_count = sum(1 for q, r in grading_results.items() 
                            if q != 'summary' and r.get('grade_category') == 'excellent')
        good_count = sum(1 for q, r in grading_results.items() 
                        if q != 'summary' and r.get('grade_category') == 'good')
        poor_count = sum(1 for q, r in grading_results.items() 
                        if q != 'summary' and r.get('grade_category') in ['poor', 'very_poor'])
        
        return f"""You are a teacher providing overall feedback on a student's exam performance.

Performance Summary:
- Overall Score: {total_percentage:.1f}%
- Questions Attempted: {questions_attempted}/{total_questions}
- Excellent Answers: {excellent_count}
- Good Answers: {good_count}
- Poor Answers: {poor_count}

Please provide encouraging overall feedback (3-4 sentences) that:
1. Acknowledges their strengths
2. Identifies areas for improvement
3. Provides motivation for future learning
4. Gives general study suggestions

Keep the tone positive and constructive."""

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without API calls."""
    
    def generate_response(self, prompt: str) -> str:
        """Generate contextual mock response based on grading information."""
        prompt_lower = prompt.lower()
        
        # Extract grading information from the prompt
        similarity_score = 0.0
        marks_awarded = 0
        max_marks = 10
        
        # Try to extract similarity score
        import re
        similarity_match = re.search(r'similarity[^\d]*([0-9]?\.[0-9]+)', prompt_lower)
        if similarity_match:
            similarity_score = float(similarity_match.group(1))
        
        # Try to extract marks
        marks_match = re.search(r'marks awarded[^\d]*(\d+)[^\d]*(\d+)', prompt_lower)
        if marks_match:
            marks_awarded = int(marks_match.group(1))
            max_marks = int(marks_match.group(2))
        
        # Calculate percentage
        percentage = (marks_awarded / max_marks * 100) if max_marks > 0 else 0
        
        if "justification" in prompt_lower:
            if similarity_score == 0.0 or "very_poor" in prompt_lower:
                return f"No answer was provided or the response was too different from the expected answer. Score: {marks_awarded}/{max_marks} ({percentage:.0f}%). Please review the question and provide a more complete answer."
            elif similarity_score < 0.3 or "poor" in prompt_lower:
                return f"The answer shows some effort but lacks key concepts and detail. Score: {marks_awarded}/{max_marks} ({percentage:.0f}%). Focus on including more relevant information."
            elif similarity_score < 0.45 or "satisfactory" in prompt_lower:
                return f"This answer covers basic concepts but could be more comprehensive. Score: {marks_awarded}/{max_marks} ({percentage:.0f}%). Good foundation with room for improvement."
            elif similarity_score < 0.6 or "good" in prompt_lower:
                return f"Good answer that demonstrates understanding of key concepts. Score: {marks_awarded}/{max_marks} ({percentage:.0f}%). Well done, with minor areas for enhancement."
            else:
                return f"Excellent answer showing comprehensive understanding. Score: {marks_awarded}/{max_marks} ({percentage:.0f}%). Demonstrates strong grasp of the subject matter."
        
        elif "improvement" in prompt_lower:
            if similarity_score == 0.0:
                return "Please provide an answer to the question. Review the course material and attempt to address all parts of the question."
            elif similarity_score < 0.3:
                return "Focus on understanding the core concepts. Review the textbook material, include more specific details, and ensure your answer directly addresses the question."
            elif similarity_score < 0.6:
                return "Good foundation. To improve: add more specific examples, explain your reasoning in greater detail, and ensure all aspects of the question are covered."
            else:
                return "Strong answer overall. Consider adding more depth to your explanations and perhaps include additional relevant examples to make your response even more comprehensive."
        
        elif "overall feedback" in prompt_lower:
            if similarity_score < 0.3:
                return "This performance indicates a need for more study and practice. Focus on understanding fundamental concepts and practicing answer structure."
            elif similarity_score < 0.6:
                return "Decent performance with room for improvement. Continue studying and focus on providing more detailed, comprehensive answers."
            else:
                return "Good performance overall. Continue practicing and refining your answers to achieve even better results."
        
        else:
            return "This is a contextual mock response based on the grading results."

def create_llm_grader(provider: str = "mock", api_key: Optional[str] = None, 
                     model: Optional[str] = None) -> LLMGrader:
    """
    Factory function to create LLM grader with appropriate provider.
    
    Args:
        provider: Provider name ("groq", "openai", "gemini", "mock")
        api_key: API key for the provider
        model: Model name to use
        
    Returns:
        Configured LLMGrader instance
    """
    if provider == "mock":
        # Create a mock grader for testing
        grader = LLMGrader.__new__(LLMGrader)
        grader.llm = MockLLMProvider()
        grader.provider_name = "mock"
        return grader
    else:
        return LLMGrader(provider, api_key, model)

def get_available_providers() -> Dict[str, Dict[str, str]]:
    """Get information about available LLM providers."""
    return {
        "groq": {
            "name": "Groq",
            "description": "Fast inference with Llama models",
            "models": "llama3-8b-8192, llama3-70b-8192",
            "free_tier": "High usage limits"
        },
        "openai": {
            "name": "OpenAI",
            "description": "GPT models from OpenAI",
            "models": "gpt-3.5-turbo, gpt-4",
            "free_tier": "Limited free credits"
        },
        "gemini": {
            "name": "Google Gemini",
            "description": "Google's latest Gemini 2.5 Flash model with excellent handwriting OCR",
            "models": "gemini-2.0-flash-exp, gemini-pro",
            "free_tier": "Generous free tier - 15 requests/minute"
        },
        "mock": {
            "name": "Mock Provider",
            "description": "Testing provider with mock responses",
            "models": "mock-model",
            "free_tier": "Always free"
        }
    }