"""
Test with explicit API version and different approach
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"Testing API Key: {api_key[:10]}...{api_key[-4:]}\n")

# Configure with explicit settings
genai.configure(api_key=api_key)

print("="*60)
print("Listing all available models...")
print("="*60)

try:
    models = genai.list_models()
    available_models = []
    
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            print(f"✓ {m.name}")
    
    if not available_models:
        print("\n❌ No models support generateContent!")
        print("\nThis means:")
        print("1. Your API key doesn't have proper permissions")
        print("2. Billing might need to be enabled")
        print("3. There might be quota/usage restrictions")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"Testing first available model: {available_models[0]}")
    print("="*60)
    
    # Try the first available model
    model = genai.GenerativeModel(available_models[0])
    response = model.generate_content("Say hello in 3 words")
    
    print("\n✅ SUCCESS!")
    print(f"Response: {response.text}")
    print(f"\n🎉 Working model: {available_models[0]}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check billing: https://console.cloud.google.com/billing")
    print("2. Check quotas: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
    print("3. Verify API key has 'Generative Language API' permission")
