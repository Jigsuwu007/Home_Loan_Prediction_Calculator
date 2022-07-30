import random
from flask import Flask, request, render_template
from markupsafe import escape
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
trainFile = pd.read_csv("/Users/Owner/Home Loan Prediction/train.csv")
trainFile['Total_Income'] = trainFile['ApplicantIncome'] + trainFile['CoapplicantIncome']
dropVariables = ['ApplicantIncome', 'CoapplicantIncome', 'Loan_ID']
trainFile = trainFile.drop(columns = dropVariables, axis = 1)
trainFile['Gender'] = trainFile['Gender'].fillna(trainFile['Gender'].mode()[0])
trainFile['Married'] = trainFile['Married'].fillna(trainFile['Married'].mode()[0])
trainFile['Dependents'] = trainFile['Dependents'].fillna(trainFile['Dependents'].mode()[0])
trainFile['Self_Employed'] = trainFile['Self_Employed'].fillna(trainFile['Self_Employed'].mode()[0])
trainFile['LoanAmount'] = trainFile['LoanAmount'].fillna(trainFile['LoanAmount'].mean())
trainFile['Loan_Amount_Term'] = trainFile['Loan_Amount_Term'].fillna(trainFile['Loan_Amount_Term'].mean())
trainFile['Credit_History'] = trainFile['Credit_History'].fillna(trainFile['Credit_History'].mean())

binaryColumns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
LE = LabelEncoder()
for columns in binaryColumns:
    trainFile[columns] = LE.fit_transform(trainFile[columns])

pd.get_dummies(trainFile['Gender'], drop_first=True)
pd.get_dummies(trainFile['Married'], drop_first=True)
pd.get_dummies(trainFile['Dependents'], drop_first=True)
pd.get_dummies(trainFile['Education'], drop_first=True)
pd.get_dummies(trainFile['Self_Employed'], drop_first=True)
pd.get_dummies(trainFile['Credit_History'], drop_first=True)
pd.get_dummies(trainFile['Property_Area'], drop_first=True)
pd.get_dummies(trainFile['Loan_Status'], drop_first=True)
pd.get_dummies(trainFile['LoanAmount'], drop_first=True)
pd.get_dummies(trainFile['Loan_Amount_Term'], drop_first=True)
pd.get_dummies(trainFile['Total_Income'], drop_first=True)

testFile = pd.read_csv("/Users/Owner/Home Loan Prediction/test.csv")
testFile['Gender'] = testFile['Gender'].fillna(testFile['Gender'].mode()[0])
testFile['Married'] = testFile['Married'].fillna(testFile['Married'].mode()[0])
testFile['Dependents'] = testFile['Dependents'].fillna(testFile['Dependents'].mode()[0])
testFile['Self_Employed'] = testFile['Self_Employed'].fillna(testFile['Self_Employed'].mode()[0])
testFile['LoanAmount'] = testFile['LoanAmount'].fillna(testFile['LoanAmount'].mean())
testFile['Loan_Amount_Term'] = testFile['Loan_Amount_Term'].fillna(testFile['Loan_Amount_Term'].mean())
testFile['Credit_History'] = testFile['Credit_History'].fillna(testFile['Credit_History'].mean())

testFile['Total_Income'] = testFile['ApplicantIncome'] + testFile['CoapplicantIncome']

binaryColumns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
LE = LabelEncoder()
for columns in binaryColumns:
    testFile[columns] = LE.fit_transform(testFile[columns])
    
dropVariables = ['ApplicantIncome', 'CoapplicantIncome', 'Loan_ID']
testFile = testFile.drop(columns = dropVariables, axis = 1)
pd.get_dummies(testFile['Gender'], drop_first=True)
pd.get_dummies(testFile['Married'], drop_first=True)
pd.get_dummies(testFile['Dependents'], drop_first=True)
pd.get_dummies(testFile['Education'], drop_first=True)
pd.get_dummies(testFile['Self_Employed'], drop_first=True)
pd.get_dummies(testFile['Credit_History'], drop_first=True)
pd.get_dummies(testFile['Property_Area'], drop_first=True)
pd.get_dummies(testFile['LoanAmount'], drop_first=True)
pd.get_dummies(testFile['Loan_Amount_Term'], drop_first=True)
pd.get_dummies(testFile['Total_Income'], drop_first=True)

trainFile = trainFile.replace(to_replace = '3+', value = 4)
testFile = testFile.replace(to_replace = '3+', value = 4)

x = trainFile.drop(columns = ['Loan_Status'], axis = 1)
y = trainFile['Loan_Status']

model = LogisticRegression()

model = pickle.load(open("model.pkl", 'rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        random_number = random.randint(10,20)*10
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state =random_number)
        gender = int(request.form['gender'])
        maritalStatus = int(request.form['maritalStatus'])
        education = int(request.form['education'])
        employmentStatus = int(request.form['employmentStatus'])
        dependents = int(request.form['dependents'])
        creditHistory = float(request.form['creditHistory'])
        propertyArea = int(request.form['propertyArea'])
        totalIncome = float(request.form['totalIncome'])
        loanAmount = float(request.form['loanAmount'])
        loanTerm = float(request.form['loanTerm'])

        model.fit(x_train, y_train)

        predictArray = np.array([[gender, maritalStatus, dependents, education, employmentStatus, loanAmount, loanTerm, creditHistory, propertyArea, totalIncome]])
        prediction = model.predict(predictArray)
        accuracy = round(model.score(x_test, y_test)*100,2)
        if (prediction == 0):
            prediction = "Unfortunately, you will not qualify for a home loan."
        elif (prediction == 1):
            prediction = "Congratulations! You will qualify for a home loan."

        return render_template("prediction.html", Prediction_Text = "{} \n Accuracy: {}%".format(prediction, accuracy))

    else:
        return render_template("prediction.html")

@app.route('/visuals')
def visuals():
    return render_template("visuals.html")
    

if __name__ == "__main__":
    app.run(debug=True)