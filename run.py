from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pandas as pd
import pickle
import os

with open("model.pkl","rb") as f:
    model = pickle.load(f)
print(os.getcwd())
app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        beer_abv = request.form["beer_abv"]
        print('beer_abv')
        review_taste = request.form["review_taste"]
        print('beer_taste')
        review_aroma = request.form["review_aroma"]
        print('beer_aroma')
        review_appearance = request.form["review_appearance"]
        print('beer_appearance')
        review_palate = request.form["review_palate"]
        print('beer_palate')
        beer_style = request.form["beer_style"]
        print('beer_style')
        X = np.array([[int(beer_style), float(review_aroma),float(review_appearance),
        float(review_palate), float(review_taste), float(beer_abv)]])
        pred = model.predict(X)[0]
        print(X)
    return render_template("index.html",pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1',port=5000)
