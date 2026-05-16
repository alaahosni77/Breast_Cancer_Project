from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')

try:
    scaler = joblib.load('models/scaler.joblib')
except:
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        
        if scaler:
            features = scaler.transform(features)
            
       
        prediction = model.predict(features)
        probability = model.predict_proba(features).tolist()
        
        result = "Malignant (0)" if prediction[0] == 0 else "Benign (1)"
        
        return jsonify({
            'prediction': int(prediction[0]),
            'result': result,
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Flask Server is running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)