import os
import uuid
import cv2
import numpy as np
import tempfile
from flask import Flask, request, redirect, url_for, render_template_string
from ultralytics import YOLO

app = Flask(__name__, static_folder="static")
os.makedirs("static/uploads",  exist_ok=True)
os.makedirs("static/results",  exist_ok=True)
model = YOLO("modelv1.onnx")

INDEX_HTML = """
<!doctype html>
<title>YOLO Detector</title>
<h2>Upload an image</h2>
<form action="/predict" method=post enctype=multipart/form-data>
  <input type=file name=file><input type=submit value=Detect>
</form>
"""

RESULT_HTML = """
<!doctype html>
<title>Result</title>
<h2>Detection result</h2>
<table>
<tr><th>Input</th><th>Output</th></tr>
<tr>
<td><img src="{{ in_path }}"  width=400></td>
<td><img src="{{ out_path }}" width=400></td>
</tr>
</table>
<br><a href="/">Try another image</a>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f or f.filename == "":
        return redirect(url_for("index"))

    uid = uuid.uuid4().hex
    ext = os.path.splitext(f.filename)[1]
    in_path = f"static/uploads/{uid}{ext}"
    out_path = f"static/results/{uid}.jpg"

    f.save(in_path)

    results = model.predict(in_path, imgsz=640, conf=0.70, device="cpu")
    annotated = results[0].plot()
    cv2.imwrite(out_path, annotated)

    return render_template_string(RESULT_HTML,
                                  in_path='/' + in_path,
                                  out_path='/' + out_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
