import numpy
import qrcode
import requests
from flask import json
import cv2

def get_webcam_image():
    cam = cv2.VideoCapture(0)
    _, img = cam.read()
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    return img_str

def send_image(img):
    files = {'file': ('img', img)}
    r = requests.post("http://miranda.rm.mt.ut.ee:5000/uploadImage", files=files)
    answer = json.loads(r.text)
    qr_img = qrcode.make(answer["url"])
    return qr_img

image = get_webcam_image()
send_image(image)
