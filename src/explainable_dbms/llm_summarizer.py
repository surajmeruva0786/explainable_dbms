import os
import google.generativeai as genai
from dotenv import load_dotenv

def summarize_text(text_to_summarize):
    """
    Summarizes the given text using the Gemini API.

    Args:
        text_to_summarize (str): The text to be summarized.

    Returns:
        str: The summarized text, or an error message if summarization fails.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Error: GEMINI_API_KEY not found in .env file."

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-pro-latest')
    print(f"Using model: {model.model_name}")

    prompt = f"textually summarize the analysations:\n\n{text_to_summarize}"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during summarization: {e}"

def save_summary(summary, filename):
    """
    Saves the summary to a text file.

    Args:
        summary (str): The summary to be saved.
        filename (str): The name of the file to save the summary to.
    """
    output_dir = "outputs/llm_summary"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(summary)
