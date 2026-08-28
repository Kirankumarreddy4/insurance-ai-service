from google import genai

client = genai.Client(
    api_key="AQ.Ab8RN6Iaxe41VRiLPUKgsNyfA9lpCpZwlJPwMPvETFHGg5akDA"
)

response = client.models.generate_content(
    model="gemini-3.7-flash",
    contents="Say Hello"
)

print(response.text)