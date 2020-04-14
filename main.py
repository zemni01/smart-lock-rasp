"""
description :
here is the main program for our project

author : Houssem Zemni
year : 2020

"""

# import commun librairies
import cv2
import numpy as np
import os
from face_recognition import recognition
import RPi.GPIO as GPIO

# set-up general pinmode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# define pins
pin_servo = 24
button=17

#set-up pin mode
GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_UP)   # add a pull-up resistor for the button 
GPIO.setup(pin_servo, GPIO.OUT)

pwm = GPIO.PWM(pin_servo, 50)     # configuration du signal PWM sur la pin du servo à une frequence de 50hz

start = 3
stop = 11

position = start      # we start from start position 

# set up the recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/facial-recognition-rasp/trainer/trainer.yml')
cascadePath = "/home/pi/facial-recognition-rasp/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

# let's define a counter for ids
id = 0

# also names related to ids ==> houssem : id=1
names = ["houssem","azmi"]

def my_callback():
    door_lock= recognition(recognizer,faceCascade)
    if door_lock == True:
        pwm.start(start)
        while True :
            if position < stop:                           # if the final position is not reached, we continue to go on 
                pwm.ChangeDutyCycle(float(position))
                position = position + 0.1
                time.sleep(0.1)
            else :
                position = start                          # when we reach the final position, we go back to the initial position
                break
            
            
# loop
while True :
    GPIO.add_event_detect(button, GPIO.RISING, callback= my_callback())               # wait for interrupt
    time.sleep(120)                                                                       #  two minutes 
        
        



