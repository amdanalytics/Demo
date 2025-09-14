from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')


# Load the pre-trained model
with open('app/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle form data from the HTML page
        if request.form:
            features = [float(request.form['feature1']), float(request.form['feature2']),
                        float(request.form['feature3']), float(request.form['feature4'])]
        # Handle JSON data from API calls
        elif request.is_json:
            data = request.get_json(force=True)
            features = data['features']
        else:
            return jsonify({'error': 'Invalid input format'}), 400

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    