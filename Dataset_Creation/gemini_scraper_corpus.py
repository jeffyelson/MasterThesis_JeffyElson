import google.generativeai as genai
import requests
genai.configure(api_key="AIzaSyAFOdPeZTcfUuwvC19yzpnT2dVw5dOclAM")

# Set up the model
generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def generate_gemini_response(url: str):
    web = driver.get(url)

    prompt_parts = "Extract the text from the website and return the text" + web
    response = model.generate_content(prompt_parts)
    return response.text

text = generate_gemini_response("https://www.jaad.org/article/S0190-9622(16)01395-5/fulltext")
print(text)