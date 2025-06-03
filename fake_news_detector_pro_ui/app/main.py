from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import os

app = FastAPI()
model_path = "model/fake_news_detector.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

class NewsInput(BaseModel):
    content: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(news: NewsInput):
    if not model:
        return {"error": "Model not found!"}
    prediction = model.predict([news.content])[0]
    return {"prediction": prediction}
