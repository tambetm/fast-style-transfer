import cv2
import qrcode
import requests
from flask import json


def get_webcam_image():
    cam = cv2.VideoCapture(0)
    _, img = cam.read()
    return img


def send_image(img, dest):
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    files = {'file': ('img', img_str)}
    r = requests.post(dest, files=files)
    answer = json.loads(r.text)
    qr_img = qrcode.make(answer["url"])
    return qr_img

# image = get_webcam_image()
# send_image(image, "http://miranda.rm.mt.ut.ee:5000/uploadImage")
