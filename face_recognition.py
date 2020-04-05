"""

description :
here is the final step for face recognition, this code will take a photo from the camera and compare its caracteristics to those already taken by the trainer in the yml file and will generate the id of the
person and how confident the recognizer is.

author : Houssem Zemni
year : 2020

"""

# import commun librairies
import cv2
import numpy as np
import os


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/facial-recognition-rasp/trainer/trainer.yml')
cascadePath = "/home/pi/facial-recognition-rasp/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

# let's define a counter for ids
id = 0

# also names related to ids ==> houssem : id=1
names = ["houssem","azmi"]

# initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)    # width
cam.set(4, 480)    # height

# let's also define minimum size of the window tobe recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True :
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH))
        )
    for (x, y, w, h) in faces :
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        
        # if confidence is less than 100 ==> "0" : perfect match
        if (confidence < 100):
            id = names[0]
            confidence = "{0}%".format(round(100 - confidence))
        else :
            id = "unknown person"
            confidence = "{0}%".format(round(100 - confidence))
            
        cv2.putText(
                    img,
                    str(id),
                    (x+5, y-5),
                    font,
                    1,
                    (255, 255, 255),
                    2
                    )
        cv2.putText(
                    img,
                    str(confidence),
                    (x+10, y+10),
                    font,
                    1,
                    (0, 0, 0),
                    1
                    )
        cv2.imshow("camera", img)
        k = cv2.waitKey(10) & 0xff
        if k == 27 :
            break
    
# clean up
print("\n [INFO] Exiting the program and clean up")
cam.release()
cv2.destroyAllWindows()
    