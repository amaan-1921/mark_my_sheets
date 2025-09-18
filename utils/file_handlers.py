import streamlit as st
import streamlit as st
from typing import Optional, Tuple, Union
import tempfile
import os

class FileHandler:
    """
    Handles file upload and processing utilities for Streamlit.
    """
    
    @staticmethod
    def save_uploaded_file(uploaded_file, directory: str = "temp") -> Optional[str]:
        """
        Save uploaded file to temporary directory.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            directory: Directory to save the file
            
        Returns:
            Path to saved file or None if failed
        """
        if uploaded_file is None:
            print("Error: uploaded_file is None")
            return None
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            print(f"Directory created/exists: {directory}")
            
            # Create temporary file with original extension
            file_extension = os.path.splitext(uploaded_file.name)[1]
            print(f"Processing file: {uploaded_file.name}, extension: {file_extension}")
            
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=file_extension, 
                dir=directory
            ) as tmp_file:
                file_content = uploaded_file.getvalue()
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
                print(f"File saved to: {tmp_path}, size: {len(file_content)} bytes")
                return tmp_path
                
        except Exception as e:
            error_msg = f"Error saving file {uploaded_file.name}: {str(e)}"
            print(error_msg)
            st.error(error_msg)
            return None
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        """
        Get file type based on extension.
        
        Args:
            filename: Name of the file
            
        Returns:
            File type string
        """
        if not filename:
            return "unknown"
        
        extension = os.path.splitext(filename)[1].lower()
        
        if extension in ['.pdf']:
            return "pdf"
        elif extension in ['.doc', '.docx']:
            return "word"
        elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            return "image"
        elif extension in ['.txt']:
            return "text"
        else:
            return "unknown"
    
    @staticmethod
    def validate_file_size(uploaded_file, max_size_mb: int = 10) -> bool:
        """
        Validate file size.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            max_size_mb: Maximum file size in MB
            
        Returns:
            True if file size is valid
        """
        if uploaded_file is None:
            return False
        
        file_size_mb = uploaded_file.size / (1024 * 1024)
        return file_size_mb <= max_size_mb
    
    @staticmethod
    def validate_file_type(uploaded_file, allowed_types: list) -> bool:
        """
        Validate file type.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            allowed_types: List of allowed file types
            
        Returns:
            True if file type is valid
        """
        if uploaded_file is None:
            return False
        
        file_type = FileHandler.get_file_type(uploaded_file.name)
        return file_type in allowed_types
    
    @staticmethod
    def cleanup_temp_files(directory: str = "temp") -> None:
        """
        Clean up temporary files.
        
        Args:
            directory: Directory to clean up
        """
        try:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up temporary files: {str(e)}")

class SessionManager:
    """
    Manages Streamlit session state for the application.
    """
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables."""
        if 'answer_key' not in st.session_state:
            st.session_state.answer_key = None
        
        if 'student_answers' not in st.session_state:
            st.session_state.student_answers = None
        
        if 'grading_results' not in st.session_state:
            st.session_state.grading_results = None
        
        if 'similarity_method' not in st.session_state:
            st.session_state.similarity_method = "cosine"
        
        if 'llm_provider' not in st.session_state:
            st.session_state.llm_provider = "mock"
        
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
        
        if 'grading_complete' not in st.session_state:
            st.session_state.grading_complete = False
    
    @staticmethod
    def clear_results():
        """Clear grading results from session state."""
        st.session_state.grading_results = None
        st.session_state.grading_complete = False
    
    @staticmethod
    def save_answer_key(answer_key):
        """Save answer key to session state."""
        st.session_state.answer_key = answer_key
        SessionManager.clear_results()
    
    @staticmethod
    def save_student_answers(student_answers):
        """Save student answers to session state."""
        st.session_state.student_answers = student_answers
        SessionManager.clear_results()
    
    @staticmethod
    def save_grading_results(results):
        """Save grading results to session state."""
        st.session_state.grading_results = results
        st.session_state.grading_complete = True

def display_file_upload_section(title: str, file_types: list, 
                               help_text: Optional[str] = None, accept_multiple_files: bool = False) -> Optional[object]:
    """
    Display file upload section with validation.
    
    Args:
        title: Title for the upload section
        file_types: List of allowed file types
        help_text: Help text to display
        accept_multiple_files: Whether to accept multiple files
        
    Returns:
        Uploaded file object(s) or None
    """
    st.subheader(title)
    
    if help_text:
        st.info(help_text)
    
    # Create file type extensions for streamlit
    extensions = []
    for file_type in file_types:
        if file_type == "pdf":
            extensions.extend([".pdf"])
        elif file_type == "word":
            extensions.extend([".doc", ".docx"])
        elif file_type == "image":
            extensions.extend([".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"])
        elif file_type == "text":
            extensions.extend([".txt"])
    
    uploaded_files = st.file_uploader(
        f"Choose file(s)" if accept_multiple_files else "Choose a file",
        type=extensions,
        key=f"upload_{title.replace(' ', '_').lower()}",
        accept_multiple_files=accept_multiple_files
    )
    
    if uploaded_files is not None:
        # Handle single file
        if not accept_multiple_files:
            # Validate file size
            if not FileHandler.validate_file_size(uploaded_files, max_size_mb=10):
                st.error("File size exceeds 10MB limit. Please upload a smaller file.")
                return None
            
            # Validate file type
            if not FileHandler.validate_file_type(uploaded_files, file_types):
                st.error(f"Invalid file type. Allowed types: {', '.join(file_types)}")
                return None
            
            # Display file info
            st.success(f"✅ Uploaded: {uploaded_files.name} ({uploaded_files.size / 1024:.1f} KB)")
            return uploaded_files
        
        # Handle multiple files
        else:
            if len(uploaded_files) == 0:
                return None
                
            valid_files = []
            total_size = 0
            
            for uploaded_file in uploaded_files:
                # Validate file size
                if not FileHandler.validate_file_size(uploaded_file, max_size_mb=10):
                    st.error(f"File {uploaded_file.name} exceeds 10MB limit. Skipping this file.")
                    continue
                
                # Validate file type
                if not FileHandler.validate_file_type(uploaded_file, file_types):
                    st.error(f"Invalid file type for {uploaded_file.name}. Allowed types: {', '.join(file_types)}")
                    continue
                
                valid_files.append(uploaded_file)
                total_size += uploaded_file.size
            
            if valid_files:
                st.success(f"✅ Uploaded {len(valid_files)} file(s) ({total_size / 1024:.1f} KB total)")
                for file in valid_files:
                    st.write(f"  • {file.name} ({file.size / 1024:.1f} KB)")
                return valid_files
            else:
                st.error("No valid files uploaded.")
                return None
    
    return None

def display_error_message(message: str, error_type: str = "error"):
    """
    Display formatted error message.
    
    Args:
        message: Error message to display
        error_type: Type of error ("error", "warning", "info")
    """
    if error_type == "error":
        st.error(f"❌ {message}")
    elif error_type == "warning":
        st.warning(f"⚠️ {message}")
    elif error_type == "info":
        st.info(f"ℹ️ {message}")

def display_success_message(message: str):
    """
    Display formatted success message.
    
    Args:
        message: Success message to display
    """
    st.success(f"✅ {message}")

def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> str:
    """
    Create download link for data.
    
    Args:
        data: Data to download
        filename: Filename for download
        mime_type: MIME type of the data
        
    Returns:
        Download link HTML
    """
    import base64
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href