import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__) # app created

# Load the pickle Model
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    # Get input values from the form
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]

    # Predict the outcome
    prediction = model.predict(features)[0]  # Extract the first element from the array

    # Map the prediction to "Approved" or "Not Approved"
    result = "Approved" if prediction == 1 else "Not Approved"

    # Render the result
    return render_template("result.html", prediction_text=f"The Loan is {result}")


if __name__ == "__main__":
    app.run(debug=True)