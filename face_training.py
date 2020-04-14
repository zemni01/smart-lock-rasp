"""

description :
in this script, we will train the opencv-recognizer with the dataset that we collect from the previous code. this code will return an YML file that will be saved on Trainer subdirectory.

author : Houssem Zemni
year : 2020

"""

# import commun librairies
import cv2
import os
import numpy as np
from PIL import Image


# path for face image database
path = "/home/pi/facial-recognition-rasp/dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/home/pi/facial-recognition-rasp/haarcascades/haarcascade_frontalface_default.xml")



# let's write a function that will bring the images and label data
def getImagesAndLabel(path):
    imagePaths = [os.path.join(path, i) for i in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths :
        PIL_image = Image.open(imagePath).convert("L")  # grayscale
        img_numpy = np.array(PIL_image, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0].split("_")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces :
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids


# let's print a message to the user 
print("\n [INFO] Training Faces. this will take a few minutes.Wait ...")


faces, ids = getImagesAndLabel(path)
recognizer.train(faces, np.array(ids))

# save the model into trainer/trainer.yml
recognizer.write("/home/pi/facial-recognition-rasp/trainer/trainer.yml")

# now, let's print the number of faces trained and end the process
print('\n [INFO] {} faces trained. Exiting Program'.format(len(np.array(ids))))
