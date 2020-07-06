import numpy as np
import cv2
import time
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_lefteye_2splits.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
#recognizer.read("zgender.yml")

labels = {"person_name":1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

kelamin = {"person_gender":1}
#with open("zjenis.pickle", 'rb') as t:
    #og_kelamin = pickle.load(t)
    #kelamin = {v:k for k,v in og_kelamin.items()}

video = cv2.VideoCapture(0)
frame_per_second = 24.0
video.set(3, 640)
video.set(4, 480)
video.set(10, 100)
id_=0
jenis_=0
count=0

log.info("Date Time, student name \n")

while True:
    check, frame = video.read()
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #canny   = cv2.Canny(frame, 100,100)
    faces   = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    count+=1

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        #jenis_, conf = recognizer.predict(roi_gray)

        if (conf>=45):
            #print(id_)
            #print(jenis_)
            print(labels[id_])
            #print(kelamin[jenis_])
            conf = " {0}%".format(round(100 - conf))
            log.info(str(dt.datetime.now()) + ","+str(id_)+"\n")
            #log.info(str(dt.datetime.now()) + ","+str(jenis_)+"\n")
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            #gender = kelamin[jenis_]
            color = (255, 0, 0)
            stroke = 1
            
            cv2.putText(frame, name, (x+160,y+10), font, stroke, color, 1, cv2.LINE_AA)
            #cv2.putText(frame,gender, (x+160,y+40), font, stroke, color, 1, cv2.LINE_AA)
            cv2.putText(frame, conf, (x+145,y+80), font, stroke, color, 1, cv2.LINE_AA)
            cv2.putText(frame, "Acces verification", (x+160,y+120),cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
        
        else:
            id = "unknown"
            conf = " {0}%".format(round(100 - conf))
            cv2.putText(frame, name, (x+160,y+10), font, 1, color, 1, cv2.LINE_AA)
            cv2.putText(frame, conf, (x+160,y+50), font, stroke, color, 1, cv2.LINE_AA)

        img_item = "my-image.jpg"
        img_item1 = "my_image.jpg"
        cv2.imwrite(img_item, roi_gray)
        cv2.imwrite(img_item1, roi_color)
        
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        
        #smiles = smile_cascade.detectMultiScale(roi_gray)
        #for (ax,ay,aw,ah) in smiles:
            #cv2.rectangle(roi_color,(ax,ay),(ax+aw,ay+ah),(0,255,0),2)

    cv2.imshow('frame',frame)
    #cv2.imshow('face-detection',gray)
    #cv2.imshow('canny', canny)
    key = cv2.waitKey(20)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()