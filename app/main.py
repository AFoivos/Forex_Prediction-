from fastapi import FastAPI

from app.routers import (
    data_handle_router,
    features_router,
    analysis_router,
    models_router,
    plots_router,
)

app = FastAPI(title="Forex Analysis API")

app.include_router(
    data_handle_router.router, 
    prefix = "/api/v1/data", 
    tags = ["Data"]
)
app.include_router(
    features_router.router, 
    prefix = "/api/v1/features", 
    tags = ["Features"]
)
app.include_router(
    analysis_router.router, 
    prefix = "/api/v1/signals", 
    tags = ["Signals"]
)
app.include_router(
    models_router.router, 
    prefix = "/api/v1/analysis", 
    tags = ["Analysis"]
)
app.include_router(
    plots_router.router, 
    prefix = "/api/v1/predict", 
    tags = ["Predictions"]
)

@app.get("/")
def root():
    return {"message": "Forex Analysis API Running!"}
