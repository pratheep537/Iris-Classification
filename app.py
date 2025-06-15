from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("decision_tree_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sl = float(request.form["sepal_length"])
        sw = float(request.form["sepal_width"])
        pl = float(request.form["petal_length"])
        pw = float(request.form["petal_width"])
        data = np.array([[sl, sw, pl, pw]])
        prediction = model.predict(data)
        species = le.inverse_transform(prediction)[0]
        return render_template("index.html", prediction=species)
    except:
        return render_template("index.html", prediction="Invalid input.")

if __name__ == "__main__":
    app.run(debug=True)
