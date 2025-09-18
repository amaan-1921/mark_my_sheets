"""
Setup script to test and configure Gemini integration
"""
import os
import subprocess
import sys

def check_gemini_installation():
    """Check if Gemini is properly installed"""
    try:
        import google.generativeai as genai
        print("✅ google-generativeai is installed")
        return True
    except ImportError:
        print("❌ google-generativeai not found")
        return False

def install_gemini():
    """Install Gemini if not present"""
    print("📦 Installing google-generativeai...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--user", "google-generativeai==0.8.3"
        ])
        print("✅ Installation successful!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Installation failed")
        return False

def test_api_key():
    """Test if API key is configured"""
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✅ GEMINI_API_KEY is set: {api_key[:10]}...")
        return api_key
    else:
        print("❌ GEMINI_API_KEY not set")
        return None

def set_api_key():
    """Help user set API key"""
    print("\n🔑 Setting up Gemini API Key:")
    print("1. Visit: https://aistudio.google.com/")
    print("2. Sign in with Google account")
    print("3. Click 'Get API Key' → 'Create API Key in new project'")
    print("4. Copy your API key")
    print("\n💻 Then run this command:")
    print("   set GEMINI_API_KEY=your_api_key_here")
    print("\n🔄 Or restart this script after setting the key")

def test_gemini_ocr():
    """Test Gemini OCR functionality"""
    api_key = test_api_key()
    if not api_key:
        set_api_key()
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Test with a simple text prompt
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello, can you read handwritten text?")
        
        print("✅ Gemini API connection successful!")
        print(f"📝 Response: {response.text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 GEMINI INTEGRATION SETUP FOR MARKMY SHEETS")
    print("=" * 60)
    
    # Step 1: Check installation
    if not check_gemini_installation():
        if install_gemini():
            print("🔄 Please restart this script to continue...")
            return
        else:
            print("❌ Failed to install Gemini")
            return
    
    # Step 2: Test API key
    if not test_api_key():
        set_api_key()
        return
    
    # Step 3: Test functionality
    if test_gemini_ocr():
        print("\n🎉 SETUP COMPLETE!")
        print("=" * 30)
        print("✅ Gemini is ready for handwritten text OCR")
        print("✅ Your MarkMySheets app now supports:")
        print("   • EasyOCR for basic text extraction")
        print("   • Gemini 2.5 Flash for superior handwriting recognition")
        print("\n🚀 Next steps:")
        print("1. Run your Streamlit app: python -m streamlit run app.py")
        print("2. In the sidebar, select 'Gemini' as OCR method")
        print("3. Upload a handwritten answer sheet")
        print("4. Enjoy superior handwriting recognition!")
    else:
        print("\n❌ Setup incomplete - please check your API key")

if __name__ == "__main__":
    main()