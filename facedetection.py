import cv2
import numpy as np

cap=cv2.VideoCapture(0)

faces=cv2.CascadeClassifier("/Users/nihalpandey/Downloads/MACHINELEAERNINGONLINE/6. Project - Face Recognition/Face Recognition Project/haarcascade_frontalface_alt.xml")

while True:
    ret,frame=cap.read()
    
    if ret==False:
        continue
    face=faces.detectMultiScale(frame,1.05,5)
    
    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow("faces",frame)
        
    
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
