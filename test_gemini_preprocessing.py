"""
Test script for Gemini OCR with advanced image preprocessing
"""
import os
import sys

def test_gemini_preprocessing():
    """Test Gemini OCR with and without preprocessing"""
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("💡 Run: $env:GEMINI_API_KEY='your_api_key_here'")
        return
    
    try:
        from modules.gemini_processor import GeminiMarkMySheets
        
        # Test sample image
        sample_path = "samples/sample1.png"
        if not os.path.exists(sample_path):
            print(f"❌ Sample file not found: {sample_path}")
            return
        
        print("🧪 TESTING GEMINI WITH ADVANCED PREPROCESSING")
        print("=" * 60)
        
        # Test WITH preprocessing (default)
        print("\n🖼️ Testing WITH advanced preprocessing:")
        gemini_with_prep = GeminiMarkMySheets(api_key, enable_preprocessing=True)
        questions_with_prep = gemini_with_prep.process_answer_sheet(sample_path)
        
        print(f"✅ Extracted {len(questions_with_prep)} questions with preprocessing")
        if questions_with_prep:
            for q_num, answer in list(questions_with_prep.items())[:2]:  # Show first 2
                print(f"  Q{q_num}: {answer[:80]}...")
        
        # Test WITHOUT preprocessing
        print("\n📷 Testing WITHOUT preprocessing:")
        gemini_no_prep = GeminiMarkMySheets(api_key, enable_preprocessing=False)
        questions_no_prep = gemini_no_prep.process_answer_sheet(sample_path)
        
        print(f"✅ Extracted {len(questions_no_prep)} questions without preprocessing")
        if questions_no_prep:
            for q_num, answer in list(questions_no_prep.items())[:2]:  # Show first 2
                print(f"  Q{q_num}: {answer[:80]}...")
        
        # Compare results
        print("\n📊 COMPARISON RESULTS:")
        print(f"With preprocessing: {len(questions_with_prep)} questions")
        print(f"Without preprocessing: {len(questions_no_prep)} questions")
        
        if len(questions_with_prep) >= len(questions_no_prep):
            print("🎉 Preprocessing improved or maintained question detection!")
        else:
            print("ℹ️ Results may vary - preprocessing optimizes for text clarity")
        
        print("\n✅ Test completed successfully!")
        print("🔧 Advanced image preprocessing is now available for your project")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure google-generativeai is installed")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_gemini_preprocessing()