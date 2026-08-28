from google import genai

client = genai.Client(
    api_key="AQ.Ab8RN6Jh0ibqvZ2n-Lh9iYiaTvWddSU9I0bW3LYSzFdoVDQC6Q"
)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say Hello"
)

print(response.text)