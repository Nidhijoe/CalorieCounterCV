import google.generativeai as genai

genai.configure(api_key="AIzaSyDlleL39YTzqQAO9M6KdHJeICCfq-BCZMk")

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Say hello")

print(response.text)