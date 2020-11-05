from flask import Flask, render_template, request
import os
import pickle
import sklearn
import pandas as pd

app = Flask(__name__)

pclass = ['f','s','t'] #클래스 등급
sex = ['f','m'] #성별
age_band = ['a','b','c','e','s','t','y'] #나이
info = [pclass, sex, age_band]
name = ['pclass', 'sex', 'age']

f = open("static/titanic_model.pkl", 'rb')
model = pickle.load(f)
f.close()

# 메인페이지
@app.route('/')
def index():
    return render_template('index.html', info = info, name = name)

# 예측페이지
@app.route("/pred", methods=['POST'])
def predict():
    if request.method == 'POST':
        result = request.form
        pred_data = make_data(result)
        #print(pred_data) #받은 데이터 확인
        pred = model.predict(pred_data)
        if  pred == 1: #생존
            res = 'The chances of this man''s survival are high.'
        elif pred == 0: #사망
            res = 'The chances of this man''s survival are low.'
        else: #오류
            print('error')
    return render_template("generic.html", res = res)

def make_data(result):
    data = [[0,0,0]]
    check = [result['pclass'], result['sex'], result['age']]
    for i in range(len(info)):
        for j in range(len(info[i])):
            if check[i] == info[i][j]:
                if i == 0:
                    if check[i]=='f':
                        data[0][i] = 1
                    elif check[i]=='s':
                        data[0][i] = 2
                    elif check[i] == 't':
                        data[0][i] = 3
                elif i == 1:
                    if check[i]=='f':
                        data[0][i] = 0
                    elif check[i]=='m':
                        data[0][i] = 1
                elif i == 2:
                    if check[i]=='y':
                        data[0][i] = 6
                    elif check[i]=='a':
                        data[0][i] = 0
                    elif check[i] == 's':
                        data[0][i] = 4
                    if check[i]=='t':
                        data[0][i] = 5
                    elif check[i]=='b':
                        data[0][i] = 1
                    elif check[i] == 'c':
                        data[0][i] = 2
                    if check[i]=='e':
                        data[0][i] = 3
    return data

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

