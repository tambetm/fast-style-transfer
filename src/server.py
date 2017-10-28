import argparse
import os
import uuid

import cv2
import numpy as np
import scipy.misc
from flask import Flask, request, flash, redirect, jsonify, send_from_directory

parser = argparse.ArgumentParser(description='Image server')
parser.add_argument('-h', "--host", help="Host ip", required=False)
parser.add_argument('-p', "--port", help="Port number", required=False)
args = parser.parse_args()

app = Flask(__name__)

UPLOAD_FOLDER = os.getcwd() + "/output"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'


@app.route("/uploadImage", methods=['POST'])
def upload_image():
    filename = str(uuid.uuid4()) + ".jpg"
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file'].read()
    image = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scipy.misc.imsave(app.config["UPLOAD_FOLDER"] + os.sep + filename, image_rgb)

    return jsonify(url=request.url_root + "downloadImage?filename=" + filename)


@app.route("/downloadImage", methods=["GET"])
def download_image():
    filename = request.args.get("filename")
    return send_from_directory(directory=app.config["UPLOAD_FOLDER"], filename=filename)


if args.host and args.port:
    app.run(args.host, args.port)
else:
    app.run()
