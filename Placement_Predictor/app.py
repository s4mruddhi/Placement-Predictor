from flask import Flask, render_template, request
import pickle
import numpy as np
# i have made this chnages.. my name sam!!!!
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
# i am ediring 
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    # Convert into array
    features = np.array([[cgpa, iq]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        result = "Placement Hoga 😄"
    else:
        result = "Placement Nahi Hoga 😢"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
