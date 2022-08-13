from flask import Flask, render_template, request, jsonify
from flask_cors import CORS,cross_origin
import numpy as np
from pickle import load

app=Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def index():
    try:
            #  reading the inputs given by the user
            CRIM=float(request.form['CRIM'])
            ZN = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])
            CHAS = float(request.form['CHAS'])
            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])
            DIS = float(request.form['DIS'])
            RAD = float(request.form['RAD'])
            TAX = float(request.form['TAX'])
            PTRATIO = float(request.form['PTRATIO'])
            B = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])

            x_input=[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]

            for i in range(0,len(x_input)):#log transform
                if x_input[i]!=0:
                    x_input[i]=np.log(x_input[i])
            
            scalefile = 'scaler.pkl'
            loaded_scaler = load(open(scalefile, 'rb')) # loading scaled model file            
            x_input_scaled=loaded_scaler.transform([x_input])
            
            filename = 'model.pkl'
            loaded_model = load(open(filename, 'rb')) # loading the model file
            # predictions using the loaded model file
            
            prediction=loaded_model.predict(x_input_scaled)
            prediction=np.exp(prediction)[0]

            # showing the prediction results in a UI
            return render_template('results.html',prediction=np.round(prediction,1))
    except Exception as e:
            print('The Exception message is: ',e)
            return 'Something went wrong'

if __name__ == "__main__":
    app.run(debug=True)