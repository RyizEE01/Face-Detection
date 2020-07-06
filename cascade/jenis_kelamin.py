import cv2
import os
import  numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
kelamin_dir = os.path.join(BASE_DIR, "gender")

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_jenis = 0
jenis_ids = {}
y_jenis = []
x_gender = []

for root, dirs, files in os.walk(kelamin_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            gender = os.path.basename(root).replace(" ", "-").lower()
            
            if not gender in jenis_ids:
                jenis_ids[gender] = current_jenis
                current_jenis += 1
            jenis_ = jenis_ids[gender]

            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            
            faces   = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                print(x, y, w, h)
                rio = image_array[y:y+h, x:x+w]
                x_gender.append(rio)
                y_jenis.append(jenis_)


with open("zjenis.pickle", 'wb') as t:
    pickle.dump(jenis_ids, t)

recognizer.train(x_gender, np.array(y_jenis))
recognizer.save("zgender.yml")