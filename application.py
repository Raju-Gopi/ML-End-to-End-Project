import sys
import os
from scr.exception import CustomException
from scr.pipeline.predictionPipeline import customData, pridictionPipeline
from flask import Flask, request, render_templates

application = Flask(__name__)
app = application

@app.route('/')
def index(self):
    return render_templates("index.html")

@app.route("/prediction", methods=["GET", "POST"])
def predict_datapoint(self):
    if request.method == "GET":
        return render_templates("home.html")
    else:
        data = customData(gender=request.form.get("gender"),
                    race_ethnicity=request.form.get('ethnicity'),
                    parental_level_of_education=request.form.get('parental_level_of_education'),
                    lunch=request.form.get('lunch'),
                    test_preparation_course=request.form.get('test_preparation_course'),
                    reading_score=float(request.form.get('writing_score')),
                    writing_score=float(request.form.get('reading_score'))
                )
        data_df = data.get_data_as_df(data)
        predict_pipe = pridictionPipeline()
        results = predict_pipe.predict(data_df)
        return render_templates("home.html", results)

if __name__ == "__main__":
    app.run("0.0.0.0")