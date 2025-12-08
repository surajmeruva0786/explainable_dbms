"""
Simple direct test of Gemini API with multiple model attempts
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"API Key: {api_key[:10]}...{api_key[-4:]}\n")

genai.configure(api_key=api_key)

# List of free models to try
models_to_try = [
    'gemini-1.5-flash-8b',
    'gemini-1.5-flash',
    'gemini-1.5-flash-latest',
    'gemini-flash-1.5',
    'gemini-pro',
]

for model_name in models_to_try:
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello in 3 words")
        print(f"✅ SUCCESS!")
        print(f"Response: {response.text}")
        print(f"\n🎉 WORKING MODEL: {model_name}")
        break
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}")
        continue
else:
    print("\n" + "="*60)
    print("❌ ALL MODELS FAILED")
    print("="*60)
    print("\nYour API key cannot access generateContent.")
    print("\nPossible solutions:")
    print("1. Enable 'Generative Language API' at:")
    print("   https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
    print("2. Check if billing is required (even for free tier)")
    print("3. Try creating API key from Google Cloud Console instead of AI Studio")
