from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Heart Disease Prediction API"
    assert body["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["label"] in ["no_disease", "disease_present"]
    assert 0 <= body["probability_no_disease"] <= 1
    assert 0 <= body["probability_disease_present"] <= 1

    probability_sum = (
        body["probability_no_disease"]
        + body["probability_disease_present"]
    )
    assert abs(probability_sum - 1.0) < 0.001


def test_predict_rejects_invalid_age():
    payload = {
        "age": 150,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422