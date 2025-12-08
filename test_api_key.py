"""
Quick test script to verify Gemini API key is working
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("="*80)
print("🔑 TESTING GEMINI API KEY")
print("="*80)

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found in .env file")
    exit(1)

print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
print("\n" + "="*80)
print("🤖 Testing connection to Gemini API...")
print("="*80)

try:
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Try gemini-1.5-flash-8b (smallest free model)
    print("\nTrying model: gemini-1.5-flash-8b")
    model = genai.GenerativeModel('gemini-1.5-flash-8b')
    
    # Send a simple test prompt
    response = model.generate_content("Say 'Hello! API is working!' in one sentence.")
    
    print("\n" + "="*80)
    print("✅ SUCCESS! API KEY IS WORKING")
    print("="*80)
    print(f"Response: {response.text}")
    print("="*80)
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ ERROR: API key test failed")
    print("="*80)
    print(f"Error: {e}")
    print("\n" + "="*80)
    print("🔄 Trying alternative model: gemini-1.5-flash")
    print("="*80)
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'Hello! API is working!' in one sentence.")
        
        print("\n" + "="*80)
        print("✅ SUCCESS! API KEY IS WORKING (with gemini-1.5-pro)")
        print("="*80)
        print(f"Response: {response.text}")
        print("="*80)
        
    except Exception as e2:
        print("\n" + "="*80)
        print("❌ BOTH MODELS FAILED")
        print("="*80)
        print(f"Error: {e2}")
        print("\nPossible solutions:")
        print("1. Get a new API key from: https://aistudio.google.com/app/apikey")
        print("2. Enable Gemini API at: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
        print("3. Check if your API key has usage limits or restrictions")
        print("="*80)
