from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from main import load_data, prepare_training_data, train_synthetic_control_model, generate_synthetic_control

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve index.html at the root
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()


# Pydantic model for SCM creation request
class SCMCreateRequest(BaseModel):
    data_source: str
    metric_column: str
    start_year: int
    end_year: int
    treatment_year: int
    input_countries: List[str]
    output_country: str


# In-memory store for generated SCMs (will be replaced by DB)
scm_store: Dict[str, Dict[str, Any]] = {}


@app.post("/create_scm")
async def create_scm(request: SCMCreateRequest):
    try:
        # Load the base data
        data_df = load_data(request.data_source, request.metric_column)

        # 1. Prepare training data
        X_train, y_train, train_years = prepare_training_data(
            data_df,
            request.start_year,
            request.treatment_year,
            request.input_countries,
            request.output_country,
            request.metric_column
        )

        # 2. Train the model to get weights
        weights = train_synthetic_control_model(X_train, y_train)

        # 3. Generate synthetic control for the full period
        synthetic_y_full, full_years = generate_synthetic_control(
            data_df,
            request.start_year,
            request.end_year,
            request.input_countries,
            weights, # Pass the original weights vector
            request.metric_column
        )

        # 4. Get factual data for the treated unit for the full period
        factual_df = data_df[
            (data_df['Year'].isin(full_years)) &
            (data_df['Entity'] == request.output_country)
        ]
        factual_df = factual_df.set_index('Year').reindex(full_years)
        factual_y_full = factual_df[request.metric_column].values

        # Prepare response data
        response_data = {
            "parameters": request.dict(),
            "years": full_years,
            "factual_metric": factual_y_full.tolist(),
            "synthetic_metric": synthetic_y_full.tolist(),
            "weights": weights.tolist() # Only country weights
        }
        
        return response_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data source file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# Placeholder endpoints
@app.get("/scms")
async def list_scms():
    return {"message": "Listing saved SCMs is not yet implemented.", "saved_scms": list(scm_store.keys())}


@app.get("/scms/{scm_id}")
async def get_scm(scm_id: str):
    if scm_id in scm_store:
        return scm_store[scm_id]
    raise HTTPException(status_code=404, detail=f"SCM with ID '{scm_id}' not found.")
