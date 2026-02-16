import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "models", "ridge.pkl"), "rb") as f:
    ridge = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        X = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        scaled_data = scaler.transform(X)
        result = ridge.predict(scaled_data)[0]

        return render_template("home.html", result=round(float(result), 3))

    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
