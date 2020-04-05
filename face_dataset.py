"""
descripion :
this scprit will allow us top create a dataset of images which we will use to train the model.

author : Houssem Zemni
year : 2020

"""
# import librairies
import cv2
import os


# initialize the camera 
cam = cv2.VideoCapture(0)
cam.set(3, 640)     # set video width
cam.set(4, 480)     # set video height
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# we will set an ID number for each person
face_id = input('\n Enter user ID and press <enter> ==> ')
print('\n [INFO] Initializing face capture, Look at the camera and wait ...')

# Now, we initialize individual sampling face count
count = 0
while True :
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces :
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        # here we should save the captured image to the dataset folder
        cv2.imwrite("/home/pi/facial-recognition-rasp/dataset/" + str(face_id) + '/' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow("image", img)
    k = cv2.waitKey(10) & 0xff  # here we can press ESC to exit the video
    if k == 27 :
        break
    elif count >= 50 :           # we will take 30 face sample and stop the video
        break
    
# let's clean up
print("\n [INFO] Exiting program and clean up Stuff")
cam.release()
cv2.destroyAllWindows()
