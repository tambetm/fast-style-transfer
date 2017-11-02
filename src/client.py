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
    #print(r.text)
    answer = json.loads(r.text)
    
    qr = qrcode.QRCode(
        box_size=3,
        border=1,
    )
    
    qr.add_data(answer["url"])
    qr.make(fit=True)
    qr_img = qr.make_image()
    return qr_img

if __name__ == '__main__':
    image = get_webcam_image()
    send_image(image, "http://magicmirror.cs.ut.ee:8080/uploadImage")
