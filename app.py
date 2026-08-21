from fastapi import FastAPI

app = FastAPI(
    title="Insurance AI Service",
    version="1.0"
)

@app.get("/")
def home():
    return {
        "message": "Insurance AI Service Running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
