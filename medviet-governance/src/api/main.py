# src/api/main.py
from fastapi import FastAPI, Depends, HTTPException
import pandas as pd
from src.access.rbac import get_current_user, require_permission
from src.pii.anonymizer import MedVietAnonymizer

app = FastAPI(title="MedViet Data API", version="1.0.0")
anonymizer = MedVietAnonymizer()


@app.get("/api/patients/raw")
@require_permission(resource="patient_data", action="read")
async def get_raw_patients(
    current_user: dict = Depends(get_current_user)
):
    """
    Tra ve raw patient data (chi admin duoc phep).
    Load tu data/raw/patients_raw.csv
    Tra ve 10 records dau tien duoi dang JSON.
    """
    df = pd.read_csv("data/raw/patients_raw.csv")
    return df.head(10).to_dict(orient="records")


@app.get("/api/patients/anonymized")
@require_permission(resource="training_data", action="read")
async def get_anonymized_patients(
    current_user: dict = Depends(get_current_user)
):
    """
    Tra ve anonymized data (ml_engineer va admin duoc phep).
    Load raw data → anonymize → tra ve JSON.
    """
    df = pd.read_csv("data/raw/patients_raw.csv")
    df_anon = anonymizer.anonymize_dataframe(df)
    return df_anon.to_dict(orient="records")


@app.get("/api/metrics/aggregated")
@require_permission(resource="aggregated_metrics", action="read")
async def get_aggregated_metrics(
    current_user: dict = Depends(get_current_user)
):
    """
    Tra ve aggregated metrics (data_analyst, ml_engineer, admin).
    So benh nhan theo tung loai benh (khong co PII).
    """
    df = pd.read_csv("data/raw/patients_raw.csv")
    metrics = df.groupby("benh").size().reset_index(name="so_luong")
    return metrics.to_dict(orient="records")


@app.delete("/api/patients/{patient_id}")
@require_permission(resource="patient_data", action="delete")
async def delete_patient(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Chi admin duoc xoa. Cac role khac nhan 403.
    """
    df = pd.read_csv("data/raw/patients_raw.csv")
    if patient_id not in df["patient_id"].values:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"message": f"Patient {patient_id} deleted", "deleted_by": current_user["username"]}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "MedViet Data API"}
