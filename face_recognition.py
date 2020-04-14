"""

description :
here is the final step for face recognition, this function will take a photo from the camera and compare its caracteristics to those already taken by the trainer in the yml file and will generate the id of the
person and how confident the recognizer is.

author : Houssem Zemni
year : 2020

"""
import cv2

def recognition(recognizer,faceCascade):
    # initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)    # width
    cam.set(4, 480)    # height

    # let's also define minimum size of the window tobe recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    
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
            door_lock = True 
        else :
            id = "unknown person"
            confidence = "{0}%".format(round(100 - confidence))
            door_lock = False
            
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
        time.sleep(60)
    
        # clean up
        print("\n [INFO] Exiting the program and clean up")
        cam.release()
        cv2.destroyAllWindows()
        return door_lock
    

    
