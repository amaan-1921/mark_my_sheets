import streamlit as st
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import os
import sys

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import custom modules
from ocr_processor import OCRProcessor, validate_questions
from answer_parser import AnswerKeyParser, QuestionMatcher
from similarity import GradingEngine, HybridGradingEngine, get_similarity_method_info
from llm_grader import create_llm_grader, get_available_providers
from file_handlers import (
    SessionManager, FileHandler, display_file_upload_section,
    display_error_message, display_success_message
)

# Page configuration
st.set_page_config(
    page_title="MarkMySheets - AI Grading Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .question-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    # Initialize session state
    SessionManager.initialize_session_state()
    
    # Initialize Gemini-only configuration
    setup_gemini_configuration()
    
    # Header
    st.markdown('<div class="main-header">MarkMySheets</div>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Automated Grading for Handwritten Answer Sheets**")
    
    # Main content
    if not st.session_state.answer_key:
        show_answer_key_upload()
    elif not st.session_state.student_answers:
        show_student_answer_upload()
    elif not st.session_state.grading_complete:
        show_grading_interface()
    else:
        show_results_dashboard()

def setup_gemini_configuration():
    """Setup Gemini-only configuration without sidebar."""
    # Force Gemini OCR method
    st.session_state.ocr_method = "gemini"
    
    # Use native extraction for answer keys by default
    st.session_state.use_ocr_answer_key = False
    
    # Gemini API key configuration
    env_api_key = os.getenv('GEMINI_API_KEY')
    
    if env_api_key:
        # Use environment variable
        st.session_state.gemini_api_key = env_api_key
       
    else:
        # Show API key input in main area
        st.warning("⚠️ Gemini API Key Required")
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password", 
            value=st.session_state.get('gemini_api_key', ''),
            help="Get your free API key from https://aistudio.google.com/. Used for intelligent PDF parsing and handwriting OCR with advanced preprocessing."
        )
        st.session_state.gemini_api_key = gemini_api_key
        
        if gemini_api_key:
            st.success("Gemini API key configured")
        else:
            st.info("Tip: Set GEMINI_API_KEY environment variable to avoid manual entry")
            return  # Don't proceed without API key
    
    # Set default similarity method (preserve existing or use fuzzy as default)
    if not hasattr(st.session_state, 'similarity_method') or not st.session_state.similarity_method:
        st.session_state.similarity_method = 'fuzzy'  # Default to fuzzy similarity for better OCR handling
    
    # Set default LLM provider to mock for now (can be changed later)
    if not hasattr(st.session_state, 'llm_provider') or not st.session_state.llm_provider:
        st.session_state.llm_provider = 'mock'
    
    # Set default API key for LLM (empty for mock)
    if not hasattr(st.session_state, 'api_key'):
        st.session_state.api_key = ''
    
    # Show configuration status
    st.info("🤖 **Configuration:** Using Gemini 2.5 Flash for OCR with advanced image preprocessing")
    
    if st.session_state.get('gemini_api_key'):
        st.success("🎆 **Intelligent Grading Enabled:** Gemini will provide conceptual understanding and educational assessment")
        st.info("📊 Similarity-based grading available as fallback for reliability")
    else:
        st.warning("⚠️ **Similarity-Only Grading:** For best results, configure Gemini API key for intelligent grading")
    
    # Add reset button in main area
    col1, col2, col3 = st.columns([1, 1, 1])
   

def show_answer_key_upload():
    """Show answer key upload interface."""
    st.markdown('<div class="section-header">Step 1: Upload Answer Key</div>', unsafe_allow_html=True)
    
    uploaded_file = display_file_upload_section(
        "Answer Key Document",
        ["pdf", "word", "image"],
        "Upload the answer key document (PDF/Word) or image containing correct answers and mark allocations."
    )
    
    if uploaded_file:
        with st.spinner("Processing answer key..."):
            try:
                # Save uploaded file
                file_path = FileHandler.save_uploaded_file(uploaded_file, "data")
                
                if file_path:
                    # Process answer key based on file type
                    file_type = FileHandler.get_file_type(uploaded_file.name)
                    
                    # Only support PDF and Word documents with Gemini
                    if file_type == "image":
                        st.error("❌ Images not supported. Please upload PDF or Word documents only.")
                        return
                    
                    # Gemini API key is required
                    gemini_api_key = st.session_state.get('gemini_api_key')
                    if not gemini_api_key:
                        st.error("❌ Gemini API key required. Please configure it in the sidebar.")
                        return
                    
                    # Initialize parser and variables
                    parser = AnswerKeyParser(gemini_api_key=gemini_api_key)
                    gemini_response = ""
                    answer_key = None
                    
                    if file_type == "pdf":
                        gemini_response = parser.extract_text_from_pdf(uploaded_file.getvalue())
                        answer_key = parser.parse_answer_key(gemini_response)
                    else:  # word
                        gemini_response = parser.extract_text_from_docx(uploaded_file.getvalue())
                        answer_key = parser.parse_answer_key(gemini_response)
                
                    
                    if answer_key:
                        SessionManager.save_answer_key(answer_key)
                        display_success_message(f"Answer key processed successfully! Found {len(answer_key)} questions.")
                        
                        # Display preview
                        show_answer_key_preview(answer_key)
                        
                        if st.button("Continue to Student Answer Upload"):
                            st.rerun()
                    else:
                        display_error_message("No questions found in the answer key. Please check the document format.")
                
            except Exception as e:
                error_msg = str(e)
                if "API failed" in error_msg:
                    display_error_message(f"❌ {error_msg}")
                else:
                    display_error_message(f"Error processing answer key: {error_msg}")

def show_student_answer_upload():
    """Show student answer upload interface with sequential processing."""
    st.markdown('<div class="section-header">Step 2: Upload Student Answer Sheet</div>', unsafe_allow_html=True)
    
    # Show answer key summary
    if st.session_state.answer_key:
        with st.expander("Answer Key Summary", expanded=False):
            show_answer_key_preview(st.session_state.answer_key, use_expander=False)
    
    # Initialize session state for sequential processing
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    if 'all_extracted_answers' not in st.session_state:
        st.session_state.all_extracted_answers = {}
    
    # Show already processed images
    if st.session_state.processed_images:
        st.subheader(f"Processed Images ({len(st.session_state.processed_images)})")
        
        cols = st.columns(min(len(st.session_state.processed_images), 3))
        for i, img_data in enumerate(st.session_state.processed_images):
            with cols[i % 3]:
                st.image(img_data['file_bytes'], caption=f"Page {i+1}: {img_data['filename']}", use_column_width=True)
                st.write(f"Found {len(img_data['answers'])} questions")
        
        # Show combined results so far
        if st.session_state.all_extracted_answers:
            with st.expander("Combined Answers So Far", expanded=False):
                for q_num, answer in sorted(st.session_state.all_extracted_answers.items()):
                    st.write(f"**Q{q_num}:**")
                    st.text(answer[:200] + "..." if len(answer) > 200 else answer)
                    st.write("---")
    
    # Upload next image
    st.subheader("Upload Next Image")
    uploaded_file = display_file_upload_section(
        f"Answer Sheet Page {len(st.session_state.processed_images) + 1}",
        ["image"],
        "Upload the next page of the student's handwritten answer sheet (PNG, JPG, etc.).",
        accept_multiple_files=False
    )
    
    if uploaded_file:
        st.subheader(f"Processing Page {len(st.session_state.processed_images) + 1}")
        st.image(uploaded_file, caption=f"Current Page: {uploaded_file.name}", use_column_width=True)
        
        if st.button("Process This Image", type="primary"):
            with st.spinner("Extracting text from current image..."):
                try:
                    # Save uploaded file
                    file_path = FileHandler.save_uploaded_file(uploaded_file, "data")
                    
                    if not file_path:
                        st.error(f"❌ Failed to save file: {uploaded_file.name}")
                        return
                    
                    st.success(f"File saved successfully: {file_path}")
                    
                    # Choose OCR method
                    ocr_method = st.session_state.get('ocr_method', 'easyocr')
                    page_answers = {}
                    
                
                    
                    if ocr_method == 'gemini' and st.session_state.get('gemini_api_key'):
                        # Use Gemini for handwriting OCR with advanced preprocessing
                        try:
                            from gemini_processor import GeminiMarkMySheets
                            
                            
                            # Enable preprocessing explicitly for optimal OCR results
                            gemini = GeminiMarkMySheets(st.session_state.gemini_api_key, enable_preprocessing=True)
                            page_answers = gemini.process_answer_sheet(file_path)
                            
                            if page_answers:
                                st.success(f"Found {len(page_answers)} questions on this page")
                            else:
                                st.warning("Found no questions on this page")
                            
                        except ImportError as e:
                            st.error(f"❌ Gemini module import failed: {e}")
                            st.info("Falling back to EasyOCR...")
                            ocr_method = 'easyocr'
                        except Exception as e:
                            st.error(f"❌ Gemini OCR error: {str(e)}")
                            st.info("Falling back to EasyOCR...")
                            ocr_method = 'easyocr'
                    
                    if ocr_method == 'easyocr':
                        # Use EasyOCR (fallback method)
                        st.info("🔍 Using EasyOCR for text extraction...")
                        
                        ocr_processor = OCRProcessor(optimize_for_handwriting=True)
                        page_answers = ocr_processor.extract_questions_from_image(file_path)
                        
                        if page_answers:
                            st.success(f"✅ EasyOCR found {len(page_answers)} questions on this page")
                        else:
                            st.warning("⚠️ EasyOCR found no questions on this page")
                    
                    # Show extracted answers from this page
                    if page_answers:
                        st.subheader("Extracted Answers from This Page")
                        for q_num, answer in sorted(page_answers.items()):
                            st.write(f"**Q{q_num}:**")
                            st.text(answer)
                            st.write("---")
                        
                        # Store this page's data
                        page_data = {
                            'filename': uploaded_file.name,
                            'file_bytes': uploaded_file.getvalue(),
                            'answers': page_answers
                        }
                        st.session_state.processed_images.append(page_data)
                        
                        # Merge with existing answers
                        for q_num, answer in page_answers.items():
                            if q_num in st.session_state.all_extracted_answers:
                                # Append to existing answer
                                st.session_state.all_extracted_answers[q_num] += f"\n\n[Continued from page {len(st.session_state.processed_images)}]\n{answer}"
                            else:
                                # New question
                                st.session_state.all_extracted_answers[q_num] = answer
                        
                        st.success(f"Page {len(st.session_state.processed_images)} processed successfully!")
                        st.info("You can now upload another image or finalize all pages below.")
                        
                        st.rerun()
                    else:
                        st.warning("No questions found on this page. You can try uploading another image or proceed with manual entry.")
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    # Finalization section
    if st.session_state.processed_images and st.session_state.all_extracted_answers:
        st.markdown("---")
        st.subheader("Finalize All Pages")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Pages Processed", len(st.session_state.processed_images))
        
        with col2:
            st.metric("Total Questions Found", len(st.session_state.all_extracted_answers))
        
        # Show final combined answers for editing
        st.subheader("Review and Edit Final Answers")
        st.info("Review the combined answers from all pages and make any necessary edits")
        
        # Validate and show editor
        validated_answers = validate_questions(st.session_state.all_extracted_answers)
        
        if validated_answers:
            edited_answers = {}
            
            for q_num in sorted(validated_answers.keys()):
                st.write(f"**Question {q_num}:**")
                edited_answer = st.text_area(
                    f"Final Answer {q_num}:",
                    value=validated_answers[q_num],
                    height=150,
                    key=f"final_answer_{q_num}"
                )
                if edited_answer.strip():
                    edited_answers[q_num] = edited_answer.strip()
                st.write("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Finalize and Continue to Grading", type="primary"):
                    if edited_answers:
                        SessionManager.save_student_answers(edited_answers)
                        display_success_message(f"Text extracted successfully! Found {len(edited_answers)} answers from {len(st.session_state.processed_images)} pages.")
                        
                        # Clear sequential processing state
                        st.session_state.processed_images = []
                        st.session_state.all_extracted_answers = {}
                        
                        st.rerun()
                    else:
                        st.error("Please provide at least one answer before proceeding.")
            
            with col2:
                if st.button("Reset and Start Over"):
                    st.session_state.processed_images = []
                    st.session_state.all_extracted_answers = {}
                    st.success("Reset complete. You can start uploading images again.")
                    st.rerun()
        else:
            st.warning("⚠️ No valid numbered questions found in the processed images.")
            
            # Manual entry option
            if st.button("Enter Answers Manually"):
                manual_answers = show_manual_answer_entry()
                if manual_answers:
                    SessionManager.save_student_answers(manual_answers)
                    display_success_message(f"Manual answers saved! Found {len(manual_answers)} answers.")
                    
                    # Clear sequential processing state
                    st.session_state.processed_images = []
                    st.session_state.all_extracted_answers = {}
                    
                    st.rerun()
    
    elif not st.session_state.processed_images:
        st.info("Upload your first image to start the sequential processing.")

def show_grading_interface():
    """Show grading interface."""
    st.markdown('<div class="section-header"> Step 3: Grade Answers</div>', unsafe_allow_html=True)
    
    # Display questions side by side in a more organized way
    if st.session_state.answer_key and st.session_state.student_answers:
        # Get all question numbers
        all_questions = sorted(set(st.session_state.answer_key.keys()) | set(st.session_state.student_answers.keys()))
        
        for q_num in all_questions:
            # Show question number with marks inline
            if q_num in st.session_state.answer_key:
                marks = st.session_state.answer_key[q_num]['marks']
                st.markdown(f"### Question {q_num} ({marks} marks)")
            else:
                st.markdown(f"### Question {q_num}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Answer Key**")
                if q_num in st.session_state.answer_key:
                    data = st.session_state.answer_key[q_num]
                    # Use text_area with disabled state for better text wrapping
                    st.text_area(
                        "Correct Answer:",
                        value=data['answer'],
                        height=150,
                        disabled=True,
                        key=f"answer_key_{q_num}"
                    )
                else:
                    st.warning("No answer key available for this question")
            
            with col2:
                st.markdown("**📝 Student Answer**")
                if q_num in st.session_state.student_answers:
                    # Use text_area with disabled state for better text wrapping
                    st.text_area(
                        "Student's Answer:",
                        value=st.session_state.student_answers[q_num],
                        height=150,
                        disabled=True,
                        key=f"student_answer_{q_num}"
                    )
                else:
                    st.warning("No student answer provided for this question")
            
            st.markdown("---")
    
    elif st.session_state.answer_key:
        st.warning("Answer key loaded but no student answers found.")
    elif st.session_state.student_answers:
        st.warning("Student answers loaded but no answer key found.")
    else:
        st.error("Both answer key and student answers are required for grading.")
    
    # Grading button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Grading", type="primary"):
            perform_grading()
    
    with col2:
        # Show recalculate button if results already exist
        if st.session_state.get('grading_results'):
            if st.button("🔄 Recalculate with Current Settings"):
                recalculate_grades()

def perform_grading():
    """Perform the main grading process using hybrid Gemini + similarity approach."""
    with st.spinner("Starting intelligent grading process..."):
        try:
            # Initialize hybrid grading engine with Gemini API key
            gemini_api_key = st.session_state.get('gemini_api_key')
            
            if gemini_api_key:
                st.info("🤖 Using Gemini intelligent grading with similarity fallback")
                grading_engine = HybridGradingEngine(
                    similarity_method=st.session_state.similarity_method,
                    gemini_api_key=gemini_api_key
                )
            else:
                st.warning("⚠️ No Gemini API key - using similarity-only grading")
                grading_engine = GradingEngine(st.session_state.similarity_method)
            
            # Grade all answers
            results = grading_engine.grade_all_answers(
                st.session_state.answer_key,
                st.session_state.student_answers
            )
            
            # Show grading method summary if hybrid was used
            if isinstance(grading_engine, HybridGradingEngine):
                summary = results.get('summary', {})
                gemini_count = summary.get('gemini_success', 0)
                similarity_count = summary.get('similarity_fallback', 0)
                
                if gemini_count > 0:
                    st.success(f"🤖 Gemini intelligent grading: {gemini_count} questions")
                if similarity_count > 0:
                    st.info(f"📊 Similarity fallback: {similarity_count} questions")
            
            # Add traditional LLM justifications only for similarity-graded answers
            # (Gemini already provides comprehensive justifications)
            similarity_graded_questions = [
                q_num for q_num, result in results.items() 
                if isinstance(q_num, int) and result.get('grading_method') != 'gemini'
            ]
            
            if similarity_graded_questions and st.session_state.llm_provider != "mock" and st.session_state.api_key:
                with st.spinner("Generating additional justifications for similarity-graded answers..."):
                    llm_grader = create_llm_grader(
                        st.session_state.llm_provider,
                        st.session_state.api_key
                    )
                    
                    for q_num in similarity_graded_questions:
                        if q_num in results:
                            result = results[q_num]
                            # Only add LLM justification if not already provided by Gemini
                            if not result.get('justification') or 'similarity' in result.get('justification', '').lower():
                                justification = llm_grader.generate_justification(
                                    q_num,
                                    result['correct_answer'],
                                    result['student_answer'],
                                    result['similarity_score'],
                                    result['marks_awarded'],
                                    result['max_marks']
                                )
                                result['justification'] = justification
                                
                                improvements = llm_grader.generate_improvement_suggestions(
                                    result['correct_answer'],
                                    result['student_answer']
                                )
                                result['improvements'] = improvements
            
            SessionManager.save_grading_results(results)
            display_success_message("🎆 Intelligent grading completed successfully!")
            st.rerun()
            
        except Exception as e:
            display_error_message(f"Error during grading: {str(e)}")

def show_results_dashboard():
    """Show results dashboard."""
    st.markdown('<div class="section-header">📊 Grading Results</div>', unsafe_allow_html=True)
    
    results = st.session_state.grading_results
    
    if not results:
        display_error_message("No grading results available.")
        return
    
    # Add recalculate button at the top
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Current Results")
    
    with col2:
        if st.button("Recalculate Grades", help="Re-grade all answers with current similarity method settings"):
            recalculate_grades()
            return
    
    with col3:
        if st.button("Back to Grading", help="Go back to modify answers before grading"):
            st.session_state.grading_complete = False
            st.rerun()
    
    # Summary metrics
    summary = results.get('summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Score", f"{summary.get('total_marks', 0)}/{summary.get('total_max_marks', 0)}")
    
    with col2:
        st.metric("Percentage", f"{summary.get('percentage', 0):.1f}%")
    
    with col3:
        st.metric("Questions Attempted", f"{summary.get('questions_attempted', 0)}/{summary.get('total_questions', 0)}")
    
    with col4:
        grade = get_letter_grade(summary.get('percentage', 0))
        st.metric("Grade", grade)
    
    # Detailed results
    show_detailed_results(results)
    
    # Charts
    show_grading_charts(results)
    
    # Overall feedback
    if 'overall_feedback' in results:
        st.subheader("Overall Feedback")
        st.info(results['overall_feedback'])
    
    # Download results
    show_download_options(results)

def show_answer_key_preview(answer_key: Dict[int, Dict[str, Any]], use_expander: bool = True):
    """Show preview of parsed answer key with complete answers."""
    st.subheader("Answer Key Preview")
    
    if not answer_key:
        st.warning("No questions found in answer key.")
        return
    
    st.success(f"Found {len(answer_key)} questions in the answer key")
    
    # Display all questions with their complete answers
    for q_num in sorted(answer_key.keys()):
        data = answer_key[q_num]
        
        st.markdown(f"### Question {q_num} ({data['marks']} marks)")
        
        # Show question text if available
        if 'question' in data and data['question']:
            st.markdown(f"**Question:** {data['question']}")
        
        # Show complete answer text - use expander only if allowed
        if use_expander:
            with st.expander(f"Complete Answer for Question {q_num}", expanded=True):
                # Use st.text to preserve formatting and show full content
                st.text(data['answer'])
        else:
            # Show answer directly without expander (for nested contexts)
            st.markdown(f"**Answer:**")
            st.text(data['answer'])
        
        st.markdown("---")

def show_student_answers_preview(student_answers: Dict[int, str]):
    """Show preview of extracted student answers."""
    st.subheader("Extracted Answers Preview")
    
    for q_num, answer in list(student_answers.items())[:3]:  # Show first 3 answers
        with st.expander(f"Question {q_num}"):
            st.write(answer)
    
    if len(student_answers) > 3:
        st.write(f"... and {len(student_answers) - 3} more answers")

def show_student_answers_editor(student_answers: Dict[int, str]) -> Optional[Dict[int, str]]:
    """Show interface for editing extracted student answers."""
    st.subheader("✏️ Edit Extracted Answers")
    st.info("💡 Review and edit the extracted answers before proceeding")
    
    edited_answers = {}
    
    for q_num in sorted(student_answers.keys()):
        st.write(f"**Question {q_num}:**")
        edited_answer = st.text_area(
            f"Answer {q_num}:",
            value=student_answers[q_num],
            height=100,
            key=f"edit_answer_{q_num}"
        )
        if edited_answer.strip():
            edited_answers[q_num] = edited_answer.strip()
        st.write("---")
    
    if st.button("✅ Confirm Edited Answers"):
        return edited_answers
    
    return None

def show_manual_answer_entry() -> Optional[Dict[int, str]]:
    """Show interface for manual answer entry."""
    st.subheader("✍️ Manual Answer Entry")
    st.info("Enter the student's answers manually")
    
    # Get number of questions from answer key
    num_questions = len(st.session_state.answer_key) if st.session_state.answer_key else 5
    
    manual_answers = {}
    
    for q_num in range(1, num_questions + 1):
        answer = st.text_area(
            f"Question {q_num} Answer:",
            height=100,
            key=f"manual_answer_{q_num}",
            help=f"Enter the student's answer for question {q_num}"
        )
        if answer.strip():
            manual_answers[q_num] = answer.strip()
    
    if st.button("✅ Save Manual Answers") and manual_answers:
        return manual_answers
    
    return None

def show_ocr_quality_assessment(raw_results: List[Dict[str, Any]]):
    """Show OCR quality assessment."""
    if not raw_results:
        return
    
    st.subheader("🔍 OCR Quality Assessment")
    
    # Calculate average confidence
    confidences = [result['confidence'] for result in raw_results if 'confidence' in result]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Text Blocks Found", len(raw_results))
    
    with col2:
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
    
    with col3:
        confidence_level = "High" if avg_confidence > 0.8 else "Medium" if avg_confidence > 0.6 else "Low"
        st.metric("Quality Level", confidence_level)
    
    # Show confidence distribution
    if confidences:
        high_conf = sum(1 for c in confidences if c > 0.8)
        medium_conf = sum(1 for c in confidences if 0.6 < c <= 0.8)
        low_conf = sum(1 for c in confidences if c <= 0.6)
        
        st.write("**Confidence Distribution:**")
        st.write(f"- High confidence (>0.8): {high_conf} blocks")
        st.write(f"- Medium confidence (0.6-0.8): {medium_conf} blocks")
        st.write(f"- Low confidence (≤0.6): {low_conf} blocks")

def show_detailed_results(results: Dict):
    """Show detailed grading results."""
    st.subheader("Detailed Results")
    
    for q_num in sorted([k for k in results.keys() if isinstance(k, int)]):
        result = results[q_num]
        
        with st.expander(f"Question {q_num} - {result['marks_awarded']}/{result['max_marks']} marks"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Correct Answer:**")
                st.write(result['correct_answer'])
                
                st.write("**Student Answer:**")
                st.write(result['student_answer'] if result['student_answer'] else "*No answer provided*")
            
            with col2:
                st.metric("Similarity Score", f"{result['similarity_score']:.2f}")
                st.metric("Grade Category", result['grade_category'].title())
                
                if 'justification' in result:
                    st.write("**AI Justification:**")
                    st.info(result['justification'])
                
                if 'improvements' in result:
                    st.write("**Suggestions for Improvement:**")
                    st.warning(result['improvements'])

def show_grading_charts(results: Dict):
    """Show grading visualization charts."""
    st.subheader("📊 Performance Analysis")
    
    # Prepare data for charts
    question_data = []
    for q_num in sorted([k for k in results.keys() if isinstance(k, int)]):
        result = results[q_num]
        question_data.append({
            'Question': f"Q{q_num}",
            'Marks Awarded': result['marks_awarded'],
            'Max Marks': result['max_marks'],
            'Percentage': result['percentage'],
            'Similarity Score': result['similarity_score'],
            'Grade Category': result['grade_category'].title()
        })
    
    df = pd.DataFrame(question_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of marks
        fig = px.bar(
            df, 
            x='Question', 
            y=['Marks Awarded', 'Max Marks'],
            title="Marks by Question",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart of grade categories
        grade_counts = df['Grade Category'].value_counts()
        fig = px.pie(
            values=grade_counts.values,
            names=grade_counts.index,
            title="Grade Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_download_options(results: Dict):
    """Show download options for results."""
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download Detailed Report"):
            report = generate_detailed_report(results)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="grading_report.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Download CSV"):
            csv_data = generate_csv_report(results)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="grading_results.csv",
                mime="text/csv"
            )

def generate_detailed_report(results: Dict) -> str:
    """Generate detailed text report."""
    summary = results.get('summary', {})
    
    report = f"""
MARKMY SHEETS - GRADING REPORT
==============================

SUMMARY:
--------
Total Score: {summary.get('total_marks', 0)}/{summary.get('total_max_marks', 0)}
Percentage: {summary.get('percentage', 0):.1f}%
Questions Attempted: {summary.get('questions_attempted', 0)}/{summary.get('total_questions', 0)}
Grade: {get_letter_grade(summary.get('percentage', 0))}

DETAILED RESULTS:
-----------------
"""
    
    for q_num in sorted([k for k in results.keys() if isinstance(k, int)]):
        result = results[q_num]
        report += f"""
Question {q_num}: {result['marks_awarded']}/{result['max_marks']} marks
Similarity Score: {result['similarity_score']:.2f}
Grade Category: {result['grade_category'].title()}

Correct Answer: {result['correct_answer']}
Student Answer: {result['student_answer'] if result['student_answer'] else 'No answer provided'}

Justification: {result.get('justification', 'Not available')}
Improvements: {result.get('improvements', 'Not available')}

{'-' * 50}
"""
    
    if 'overall_feedback' in results:
        report += f"\nOVERALL FEEDBACK:\n{results['overall_feedback']}\n"
    
    return report

def generate_csv_report(results: Dict) -> str:
    """Generate CSV report."""
    data = []
    
    for q_num in sorted([k for k in results.keys() if isinstance(k, int)]):
        result = results[q_num]
        data.append([
            q_num,
            result['marks_awarded'],
            result['max_marks'],
            f"{result['percentage']:.1f}%",
            f"{result['similarity_score']:.2f}",
            result['grade_category'].title(),
            result.get('justification', '').replace('\n', ' '),
            result.get('improvements', '').replace('\n', ' ')
        ])
    
    df = pd.DataFrame(data, columns=[
        'Question', 'Marks Awarded', 'Max Marks', 'Percentage', 
        'Similarity Score', 'Grade Category', 'Justification', 'Improvements'
    ])
    
    return df.to_csv(index=False)

def recalculate_grades():
    """Recalculate grades with current settings using hybrid approach."""
    st.info("🔄 Recalculating grades with current similarity method and intelligent grading...")
    
    with st.spinner("Re-grading in progress..."):
        try:
            # Initialize hybrid grading engine with current settings
            gemini_api_key = st.session_state.get('gemini_api_key')
            
            if gemini_api_key:
                grading_engine = HybridGradingEngine(
                    similarity_method=st.session_state.similarity_method,
                    gemini_api_key=gemini_api_key
                )
            else:
                grading_engine = GradingEngine(st.session_state.similarity_method)
            
            # Grade all answers again
            results = grading_engine.grade_all_answers(
                st.session_state.answer_key,
                st.session_state.student_answers
            )
            
            # Add LLM justifications if available
            if st.session_state.llm_provider != "mock" and st.session_state.api_key:
                llm_grader = create_llm_grader(
                    st.session_state.llm_provider,
                    st.session_state.api_key
                )
            elif st.session_state.llm_provider == "mock":
                llm_grader = create_llm_grader("mock")
            else:
                llm_grader = None
            
            # Add justifications to results
            if llm_grader:
                with st.spinner("Generating updated AI justifications..."):
                    for q_num in st.session_state.answer_key.keys():
                        if q_num in results:
                            result = results[q_num]
                            justification = llm_grader.generate_justification(
                                q_num,
                                result['correct_answer'],
                                result['student_answer'],
                                result['similarity_score'],
                                result['marks_awarded'],
                                result['max_marks']
                            )
                            result['justification'] = justification
                            
                            # Generate improvement suggestions
                            improvements = llm_grader.generate_improvement_suggestions(
                                result['correct_answer'],
                                result['student_answer']
                            )
                            result['improvements'] = improvements
                    
                    # Generate overall feedback
                    overall_feedback = llm_grader.generate_overall_feedback(results)
                    results['overall_feedback'] = overall_feedback
            
            # Save updated results
            SessionManager.save_grading_results(results)
            
            # Show comparison with previous results if available
            show_recalculation_comparison(results)
            
            display_success_message("Grades recalculated successfully with updated settings!")
            st.rerun()
            
        except Exception as e:
            display_error_message(f"Error during recalculation: {str(e)}")

def show_recalculation_comparison(new_results: Dict):
    """Show comparison between old and new grading results."""
    st.subheader("🔍 Recalculation Summary")
    
    # Show method used
    st.info(f"📊 Similarity Method Used: {st.session_state.similarity_method.title()}")
    
    # Show new summary
    new_summary = new_results.get('summary', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "New Total Score", 
            f"{new_summary.get('total_marks', 0)}/{new_summary.get('total_max_marks', 0)}"
        )
    
    with col2:
        st.metric(
            "New Percentage", 
            f"{new_summary.get('percentage', 0):.1f}%"
        )
    
    # Show grade distribution
    grade_counts = {}
    for q_num in [k for k in new_results.keys() if isinstance(k, int)]:
        category = new_results[q_num]['grade_category']
        grade_counts[category] = grade_counts.get(category, 0) + 1
    
    if grade_counts:
        st.write("**Grade Distribution:**")
        for category, count in sorted(grade_counts.items()):
            st.write(f"- {category.title()}: {count} question(s)")

def get_letter_grade(percentage: float) -> str:
    """Convert percentage to letter grade."""
    if percentage >= 90:
        return "A+"
    elif percentage >= 85:
        return "A"
    elif percentage >= 80:
        return "A-"
    elif percentage >= 75:
        return "B+"
    elif percentage >= 70:
        return "B"
    elif percentage >= 65:
        return "B-"
    elif percentage >= 60:
        return "C+"
    elif percentage >= 55:
        return "C"
    elif percentage >= 50:
        return "C-"
    else:
        return "F"

def reset_application():
    """Reset the entire application state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Clean up temporary files
    FileHandler.cleanup_temp_files("data")
    
    st.success("Application reset successfully!")
    st.rerun()

if __name__ == "__main__":
    main()