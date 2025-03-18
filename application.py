import pickle
from flask import Flask , render_template , request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## Importing ridge regressor and standard scaler 
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
scaler_model = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature')) ##--> yaha sbke input le rahe ha hr ke feature ke 
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]) # Transform krke values ko ek new variable ma dalinge
        result=ridge_model.predict(new_data_scaled) ## Prediction ka kaam hora h yaha pe 

        return render_template('home.html',result=result[0]) # Home page hi waps return krna h hme and oske sath oska result aayga osme result LIST ke form ma hoga islia hme ek variable ma store krani hogi first value result ki 
    else:
        return render_template('home.html')
    




if __name__ == '__main__':
    app.run(debug=True)
