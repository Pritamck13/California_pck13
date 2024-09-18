import pickle
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Initialize FastAPI app
app = FastAPI()

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Load templates (similar to render_template in Flask)
templates = Jinja2Templates(directory="templates")


# Define a Pydantic model for the request body
class DataModel(BaseModel):
    data: dict


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Render the home.html template
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict_api")
async def predict_api(data: DataModel):
    # Get data from request and process it
    input_data = data.data
    transformed_data = scalar.transform(np.array(list(input_data.values())).reshape(1, -1))
    prediction = regmodel.predict(transformed_data)
    return {"prediction": prediction[0]}


@app.post("/predict")
async def predict_form(request: Request,MedInc:float = Form(...),
                                        HouseAge:float = Form(...),
                                        AveRooms:float = Form(...),
                                        AveBedrms:float = Form(...),
                                        Population:float = Form(...),
                                        AveOccup:float = Form(...),
                                        Latitude:float = Form(...),
                                        Longitude:float = Form(...)):
    # Process form data
    input_data = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    transformed_data = scalar.transform([input_data])
    prediction = regmodel.predict(transformed_data)[0]
    # Render prediction in the template
    return templates.TemplateResponse("home.html", {
        "request": request, 
        "prediction_text": f"The House price prediction is {prediction}"
    })



