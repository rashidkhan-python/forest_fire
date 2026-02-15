from sklearn.preprocessing import StandardScaler
from flask import Flask,jsonify,request,render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Importing pickel file (Models, StandardScaler)

ridge = pickle.load(open(r'C:\Users\Rashid\Desktop\UKDlDs\Basic ML Project\models\ridge.pkl','rb'))
scaler = pickle.load(open(r'C:\Users\Rashid\Desktop\UKDlDs\Basic ML Project\models\scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template(r"index.html")

#Route for Predict Data
@app.route('/predict',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            scaled_data = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
            result = ridge.predict(scaled_data)

            return render_template('home.html',result = round(result[0],3))

    else:
        return render_template('home.html')

if __name__ == "__main__":  
    app.run("0.0.0.0")