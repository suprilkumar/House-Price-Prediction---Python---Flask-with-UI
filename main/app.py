from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Prepare options for dropdowns
dropdown_options = {key: list(encoder.classes_) for key, encoder in encoders.items()}

# Helper function to format number in Indian currency style
def format_indian_currency(amount):
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return s
    else:
        last_three = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        return ','.join(parts) + ',' + last_three

@app.route('/')
def home():
    return render_template("index.html", options=dropdown_options)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        area = float(request.form['area'])
        bhk = int(request.form['bhk'])
        bathroom = int(request.form['bathroom'])
        furnishing = encoders["Furnishing"].transform([request.form['furnishing']])[0]
        locality = encoders["Locality"].transform([request.form['locality']])[0]
        parking = int(request.form['parking'])
        status = encoders["Status"].transform([request.form['status']])[0]
        transaction = encoders["Transaction"].transform([request.form['transaction']])[0]
        type_ = encoders["Type"].transform([request.form['type']])[0]

        features = np.array([[area, bhk, bathroom, furnishing, locality, parking, status, transaction, type_]])
        predicted_price = model.predict(features)[0]

        formatted_price = format_indian_currency(predicted_price)

        return render_template("index.html", options=dropdown_options, prediction=formatted_price)
    except Exception as e:
        return render_template("index.html", options=dropdown_options, error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5502)
