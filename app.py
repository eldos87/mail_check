from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

with open ('model','rb') as file:
    clf = pickle.load(file)

with open ('cv','rb') as f:
    cv = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        msg = request.form['message']
        data = [msg]
        transformed_msg = cv.transform(data)
        pred = clf.predict(transformed_msg)
    return render_template('result.html',prediction = pred)

if __name__ == '__main__':
    app.run()
