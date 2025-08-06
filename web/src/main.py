from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import warnings
from dotenv import load_dotenv

from routes import router
from utilitaires import init_database

warnings.filterwarnings('ignore')
load_dotenv()

app = FastAPI(title="DiabetoWeb", description="Application de prédiction du diabète")

# Configuration des templates et fichiers statiques
templates = Jinja2Templates(directory="../templates")
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Inclure les routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    init_database()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)