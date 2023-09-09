'''
Estas son las librerias que instalamos para que funcione la api
pip install flask pymongo
pip install flask-pymongo
pip install flask cors
pip install python-dotenv


Para entrar al ambiente virtual:
venv\Scripts\activate

Este no es el script bueno de la api
'''

#-----------------Esta es la inicializacion de la base de datos-------------------------
#Importamos las librerias
import os
from pickle import TRUE
from dotenv import load_dotenv 
from flask import Flask, request
from bson.objectid import ObjectId
from flask_cors import CORS
import pandas as pd

import json
import csv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__) #Inicializamos Flask
CORS(app)

df = pd.read_csv('train_clean.csv')

X = df.drop(columns = ["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'C': 2.7399744574633367, 
    'penalty': 'l2', 
    'solver': 'lbfgs', 
    'fit_intercept': True, 
    'class_weight': None, 
    'tol': 0.00020503644013261026, 
    'max_iter': 700}

model = LogisticRegression(
    C=params['C'],
    penalty=params['penalty'],
    solver=params['solver'],
    fit_intercept=params['fit_intercept'],
    class_weight=params['class_weight'],
    tol=params['tol'],
    max_iter=params['max_iter'],
    random_state=42
)

model.fit(X_train, y_train)

#--------------------------Estas son las llamadas a la base de datos--------------------------------
@app.route('/data', methods=['GET'])
def get_data():
    data_json = df.to_json(orient = 'table')

    # Parse the JSON string
    data_dict = json.loads(data_json)

    # Extract the "data" part
    data = data_dict.get("data", [])

    return data

@app.route('/predict', methods=['POST'])
def predict():
    #Usamos el body para los parametros porque son sensibles y tambien porque queremos que sean case sensitive (mientras no los mandamos con la app en hash)
    Pclass = request.json['Pclass'] 
    Sex = request.json['Sex'] 
    Age = request.json['Age'] 
    SibSp = request.json['SibSp'] 
    Parch = request.json['Parch'] 
    Fare = request.json['Fare'] 
    Embarked = request.json['Embarked'] 
    Title = request.json['Title'] 
    '''
    API call example
    {
    "Pclass": 3,
    "Sex": "Male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "Title": "Mr",
    }
    '''

    #Male is 1, Female is 0
    if Sex == 'Male':
        Sex = 1
    elif Sex == 'Female':
        Sex = 0
    else:
        Sex = 1

    #Order C, Q, S
    if Embarked == 'C':
        Embarked = [1,0,0]
    elif Embarked == 'Q':
        Embarked = [0,1,0]
    elif Embarked == 'S':
        Embarked = [0,0,1]
    else:
        Embarked = [0,0,0]

    #Master, miss, mr, mrs, officer, royal
    if Title == 'Master':
        Title = [1,0,0,0,0,0]
    elif Title == 'Miss':
        Title = [0,1,0,0,0,0]
    elif Title == 'Mr':
        Title = [0,0,1,0,0,0]
    elif Title == 'Mrs':    
        Title = [0,0,0,1,0,0]
    elif Title == 'Officer':
        Title = [0,0,0,0,1,0]
    elif Title == 'Royal':
        Title = [0,0,0,0,0,1]
    else:
        Title = [0,0,0,0,0,0]

    inputData = {
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked_C": Embarked[0],
        "Embarked_Q": Embarked[1],
        "Embarked_S": Embarked[2],
        "Title_Master": Title[0],
        "Title_Miss": Title[1],
        "Title_Mr": Title[2],
        "Title_Mrs": Title[3],
        "Title_Officer": Title[4],
        "Title_Royal": Title[5],
    }
    
    inputDf = pd.DataFrame([inputData])
    y_pred = model.predict(inputDf)
    if y_pred[0] == 0:
        return {
            'prediction': 'Survived'
        } 
    elif y_pred[0] == 1:
        return {
            'prediction': 'Died'
        } 

@app.route('/new/register', methods=['POST'])
def save_data():
    #Usamos el body para los parametros porque son sensibles y tambien porque queremos que sean case sensitive (mientras no los mandamos con la app en hash)
    Pclass = request.json['Pclass'] 
    Sex = request.json['Sex'] 
    Age = request.json['Age'] 
    SibSp = request.json['SibSp'] 
    Parch = request.json['Parch'] 
    Fare = request.json['Fare'] 
    Embarked = request.json['Embarked'] 
    Title = request.json['Title'] 
    Survived = request.json['Survived'] 
    '''
    API call example
    {
    "Survived": 1,
    "Pclass": 3,
    "Sex": "Male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "Title": "Mr"
    }
    '''

    #Male is 1, Female is 0
    if Sex == 'Male':
        Sex = 1
    elif Sex == 'Female':
        Sex = 0
    else:
        Sex = 1

    #Order C, Q, S
    if Embarked == 'C':
        Embarked = [1,0,0]
    elif Embarked == 'Q':
        Embarked = [0,1,0]
    elif Embarked == 'S':
        Embarked = [0,0,1]
    else:
        Embarked = [0,0,0]

    #Master, miss, mr, mrs, officer, royal
    if Title == 'Master':
        Title = [1,0,0,0,0,0]
    elif Title == 'Miss':
        Title = [0,1,0,0,0,0]
    elif Title == 'Mr':
        Title = [0,0,1,0,0,0]
    elif Title == 'Mrs':    
        Title = [0,0,0,1,0,0]
    elif Title == 'Officer':
        Title = [0,0,0,0,1,0]
    elif Title == 'Royal':
        Title = [0,0,0,0,0,1]
    else:
        Title = [0,0,0,0,0,0]
    
    data = [
        Survived,
        Pclass,
        Sex,
        Age,
        SibSp,
        Parch,
        Fare,
        Embarked[0],
        Embarked[1],
        Embarked[2],
        Title[0],
        Title[1],
        Title[2],
        Title[3],
        Title[4],
        Title[5]
    ]

    # Specify the CSV file path
    csv_file_path = 'train_clean.csv'

    # Open the CSV file in 'append' mode (to add a new row to the existing data)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)  # Create a CSV writer object
        writer.writerow(data)  # Write the new record as a row to the CSV file

    df = pd.read_csv('train_clean.csv')
    X = df.drop(columns = ["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(
        C=params['C'],
        penalty=params['penalty'],
        solver=params['solver'],
        fit_intercept=params['fit_intercept'],
        class_weight=params['class_weight'],
        tol=params['tol'],
        max_iter=params['max_iter'],
        random_state=42
    )
    model.fit(X_train, y_train)

    return {
        'status': 200,
        'data': 'Data saved',
        'model': 'retrained'
    }

if __name__ == "__main__":
    app.run(debug=True) #Con el True en debug se reinicia cuando hay cambios