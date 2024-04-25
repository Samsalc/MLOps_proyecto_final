from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="Deploy fetal health classification model",
    version="0.0.1"
)


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
model = joblib.load("model/random_forest_model_v01.pkl")


@app.post("/api/v1/predict-fetal-health", tags=["fetal-health"])
async def predict(
    baseline_value: float,
    accelerations: float,
    fetal_movement: float,
    uterine_contractions: float,
    light_decelerations: float,
    severe_decelerations: float,
    prolongued_decelerations: float,
    abnormal_short_term_variability: float,
    mean_value_of_short_term_variability: float,
    percentage_of_time_with_abnormal_long_term_variability: float,
    mean_value_of_long_term_variability: float,
    histogram_width: float,
    histogram_min: float,
    histogram_max: float,
    histogram_number_of_peaks: float,
    histogram_number_of_zeroes: float,
    histogram_mode: float,
    histogram_mean: float,
    histogram_median: float,
    histogram_variance: float,
    histogram_tendency: float
):
    dictionary = {
        'baseline value': baseline_value,
        'accelerations': accelerations,
        'fetal_movement': fetal_movement,
        'uterine_contractions': uterine_contractions,
        'light_decelerations': light_decelerations,
        'severe_decelerations': severe_decelerations,
        'prolongued_decelerations': prolongued_decelerations,
        'abnormal_short_term_variability': abnormal_short_term_variability,
        'mean_value_of_short_term_variability': mean_value_of_short_term_variability,
        'percentage_of_time_with_abnormal_long_term_variability': percentage_of_time_with_abnormal_long_term_variability,
        'mean_value_of_long_term_variability': mean_value_of_long_term_variability,
        'histogram_width': histogram_width,
        'histogram_min': histogram_min,
        'histogram_max': histogram_max,
        'histogram_number_of_peaks': histogram_number_of_peaks,
        'histogram_number_of_zeroes': histogram_number_of_zeroes,
        'histogram_mode': histogram_mode,
        'histogram_mean': histogram_mean,
        'histogram_median': histogram_median,
        'histogram_variance': histogram_variance,
        'histogram_tendency': histogram_tendency

    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        prediction = int(prediction[0])
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": prediction}
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )