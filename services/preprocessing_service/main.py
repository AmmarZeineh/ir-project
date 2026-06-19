from fastapi import FastAPI
from pydantic import BaseModel
from .preprocessor import preprocess

app = FastAPI(title="Preprocessing Service", version="1.0")

class TextInput(BaseModel):
    text: str
    use_stemming: bool = False
    use_lemmatization: bool = True
    remove_stops: bool = True

@app.post("/preprocess")
def preprocess_text(body: TextInput):
    return preprocess(body.text, body.use_stemming, body.use_lemmatization, body.remove_stops)

@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing", "port": 8001}