"""
Permanent API Key Setup Script for MarkMySheets
"""
import os
import subprocess
import sys

def set_permanent_api_key():
    """Set Gemini API key permanently"""
    print("🔐 PERMANENT GEMINI API KEY SETUP")
    print("=" * 50)
    
    # Check if already set
    current_key = os.getenv('GEMINI_API_KEY')
    if current_key:
        print(f"✅ Current API key: {current_key[:10]}...")
        choice = input("📝 Do you want to update it? (y/n): ").strip().lower()
        if choice != 'y':
            print("👍 Keeping current API key")
            return
    
    # Get API key from user
    print("\n🔑 Get your API key from: https://aistudio.google.com/")
    api_key = input("🔐 Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    # Validate API key format (basic check)
    if len(api_key) < 20:
        print("⚠️ API key seems too short. Are you sure it's correct?")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return
    
    try:
        # Method 1: PowerShell (User level)
        print("\n💻 Setting environment variable...")
        
        ps_command = f'[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "{api_key}", "User")'
        
        result = subprocess.run([
            "powershell", "-Command", ps_command
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ API key set successfully!")
            print("🔄 Please restart your terminal/IDE for changes to take effect")
            
            # Test by starting a new process
            test_result = subprocess.run([
                "powershell", "-Command", 'echo $env:GEMINI_API_KEY'
            ], capture_output=True, text=True)
            
            if api_key[:10] in test_result.stdout:
                print("✅ Verification successful!")
            else:
                print("⚠️ Verification failed - you may need to restart your terminal")
                
        else:
            print(f"❌ PowerShell method failed: {result.stderr}")
            print("💡 Try the manual GUI method instead")
            
    except Exception as e:
        print(f"❌ Error setting API key: {e}")
        print("\n🔧 MANUAL SETUP INSTRUCTIONS:")
        print("1. Press Win + R, type 'sysdm.cpl', press Enter")
        print("2. Click 'Environment Variables'")
        print("3. Under 'User variables', click 'New'")
        print("4. Variable name: GEMINI_API_KEY")
        print(f"5. Variable value: {api_key}")
        print("6. Click OK, OK, OK")
        print("7. Restart your terminal/IDE")

def verify_api_key():
    """Verify the API key is working"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        # Quick test with Gemini
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello")
        
        print("✅ API key is working!")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 MARKMY SHEETS - PERMANENT API KEY SETUP")
    print("=" * 60)
    
    choice = input("Choose an option:\n1. Set new API key\n2. Verify current API key\nEnter choice (1/2): ").strip()
    
    if choice == "1":
        set_permanent_api_key()
    elif choice == "2":
        verify_api_key()
    else:
        print("❌ Invalid choice")
        return
    
    print("\n🎉 Setup complete!")
    print("💡 Remember to restart your terminal/IDE after setting the API key")

if __name__ == "__main__":
    main()