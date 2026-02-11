from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.models.v1.predict import predict_top_clients  # pour l'instant V1
from src.app.utils import load_registry, get_model_metrics

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
    registry = load_registry()

    # Liste tous les modèles disponibles
    models_list = [
        sizedata["model_file"]
        for vdata in registry.values()
        for subdata in vdata["models"].values()
        for paramdata in subdata.values()
        for sizedata in paramdata.values()
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": None,
            "top_n": None,
            "model_name": None,
            "model_metrics": None,
            "models_list": models_list,
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
    model_name: str = Form("v1.2.param1.size3.joblib")  # default
):
    error = None
    results = None
    model_metrics = None

    try:
        if top_n <= 0:
            raise ValueError("top_n doit être > 0")

        # ⚡ Prédiction des top clients
        results = predict_top_clients(
            model_name=model_name,
            top_n=top_n
        )

        # ⚡ Conversion en liste sûre pour Jinja
        if isinstance(results, list):
            results_list = results
        else:
            results_list = results.to_dict(orient="records")

        if not results_list:
            raise ValueError("Aucun résultat retourné par la fonction de prédiction")

        results = results_list

        # ⚡ Charger métriques depuis registry
        registry = load_registry()
        model_metrics = get_model_metrics(model_name, registry)

    except Exception as e:
        error = str(e)

    # ⚡ Dropdown des modèles pour l'UI (toujours recalculé)
    registry = load_registry()
    models_list = [
        sizedata["model_file"]
        for vdata in registry.values()
        for subdata in vdata["models"].values()
        for paramdata in subdata.values()
        for sizedata in paramdata.values()
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "top_n": top_n,
            "model_name": model_name,
            "model_metrics": model_metrics,
            "models_list": models_list,
            "error": error
        }
    )
