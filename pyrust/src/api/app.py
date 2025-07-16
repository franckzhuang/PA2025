from fastapi import FastAPI
from pyrust.src.api.controllers import training
from pyrust.src.api.controllers import evaluating

app = FastAPI(
    title="Training API",
    description="API to launch and follow models jobs training.",
    version="1.0.0",
)

app.include_router(training.router)
app.include_router(evaluating.router)


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Pyrust Training API"}
