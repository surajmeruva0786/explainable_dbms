"""
List all available Gemini models with the current API key
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("="*80)
print("📋 LISTING AVAILABLE GEMINI MODELS")
print("="*80)

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found in .env file")
    exit(1)

print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}\n")

try:
    genai.configure(api_key=api_key)
    
    print("Available models:")
    print("-" * 80)
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description[:100]}...")
            print()
    
    print("="*80)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nThis might mean:")
    print("1. API key is invalid")
    print("2. API is not enabled")
    print("3. Network/firewall issue")
    print("\nGet a new API key from: https://aistudio.google.com/app/apikey")
