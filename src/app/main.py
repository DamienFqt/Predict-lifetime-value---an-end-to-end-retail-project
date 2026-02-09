from fastapi import FastAPI, Request, Form
import pandas as pd
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.models.v1.predict import predict_top_clients

app = FastAPI(title="CLV Prediction App")

# ⚠️ Chemin RELATIF au point de lancement uvicorn
templates = Jinja2Templates(directory="src/app/templates")


# -----------------------
# Root → redirect UI
# -----------------------
@app.get("/")
def root():
    return RedirectResponse(url="/ui")


# -----------------------
# UI - GET
# -----------------------
@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": None,
            "top_n": None,
            "model_name": None,
            "error": None
        }
    )


# -----------------------
# UI - POST (prediction)
# -----------------------
@app.post("/ui", response_class=HTMLResponse)
def ui_predict(
    request: Request,
    top_n: int = Form(...),
    model_name: str = Form("v1.2.param1.size3.joblib")  # hardcodé OK pour V1
):
    error = None
    results = None
    try:
        if top_n <= 0:
            raise ValueError("top_n doit être > 0")

        results = predict_top_clients(
            model_name=model_name,
            top_n=top_n
        )

        if results is None:
            raise ValueError("Aucun résultat retourné par la fonction de prédiction")
        
        print("TYPE results:", type(results))
        print("RESULTS:", results)

    except Exception as e:
        error = str(e)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "top_n": top_n,
            "model_name": model_name,
            "error": error
        }
    )
