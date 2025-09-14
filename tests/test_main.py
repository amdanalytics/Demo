from .. app.main import app

def test_home():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b"Iris Prediction" in response.data  # Check for page title


def test_predict():
    client = app.test_client()
    response = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert 'prediction' in response.json