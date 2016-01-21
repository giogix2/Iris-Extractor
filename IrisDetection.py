#Capture a video using the library cv2 and than it saves the last high resolution image, after pressing the button q.
#The last image is saved thank to the command "capture" of the library picamera, instead of the library cv2.
#In this script the final image is converted in an OpenCV object to be elaborated later. After the conversion of the image
#a few algorithm for face detection and eyes detection are executed, and the new images are saved.




#import numpy as np
import cv2
import picamera
import io
import sys
import numpy as np
from functions import *
import argparse

__author__ = 'Giovanni Ortolani'

#Create imput arguments
parser = argparse.ArgumentParser(description='Iris detector by Giovanni Ortolani.')
parser.add_argument('-i','--input', help='Input file name',required=False)
parser.add_argument('-o','--output',help='Output file name', required=False)
parser.add_argument('-p','--pupil',help='Output file name', required=False)
parser.add_argument('-e','--event',help='Output file name', required=False)
args = parser.parse_args()


if args.output is not None:
    filename = args.output
else:
    filename = '/home/pi/Desktop/foto'

#Variables
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/Raspberry/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/pi/Desktop/Raspberry/haarcascade_eye.xml')
resAcquisitionWidth = 160
resAcquisitionHeight = 120
resCaptureWidth = 2592
resCaptureHeight = 1944
scaleWidth = resCaptureWidth/resAcquisitionWidth
scaleHeight = resCaptureHeight/resAcquisitionHeight
eyesCycleCounter = 0
eyes = [0]
isFacePositioned = False
rectWide = 0
rectHeight = 0

#Prepare the video capturing
cap = cv2.VideoCapture(0)
cap.set(3, resAcquisitionWidth);
cap.set(4, resAcquisitionHeight);

# Create the in-memory stream
stream = io.BytesIO()
cv2.namedWindow('frame')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    t = cv2.getTickCount()
    # Detect the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    t = cv2.getTickCount() - t
    print "time taken for detection = %gms" % (t/(cv2.getTickFrequency()*1000.))
    for (x,y,w,h) in faces:
        rectWide = w
        rectHeight = h
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    if len(eyes) == 2:
    	eyesCycleCounter = eyesCycleCounter+1
    else:
    	eyesCycleCounter = 0

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if args.event == 'face':
        if eyesCycleCounter == 10 and percentage(rectWide, resAcquisitionWidth) > 51.0 and percentage(rectHeight, resAcquisitionHeight) > 68.0:
            isFacePositioned = True


    if (cv2.waitKey(1) & 0xFF == ord('q')) or isFacePositioned:
        cap.release()
        with picamera.PiCamera() as camera:
            camera.resolution = (resCaptureWidth, resCaptureHeight)
            camera.capture(stream, format='jpeg')

            print('Immagine acquisita !!')
            camera.close()
            # Construct a numpy array from the stream            
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            # "Decode" the image from the array, preserving colour. "image" becomes
            # a real opencv (cv2) object, and we can work on it with all the functions
            # avaiable in the library
            image = cv2.imdecode(data, 1)
            imageCopy = image

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            i = 0
            for (x,y,w,h) in faces:
                x = int(x*scaleWidth)
                y = int(y*scaleHeight)
                w = int(w*scaleWidth)
                h = int(h*scaleHeight)
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                for (ex,ey,ew,eh) in eyes:
                    ex = int(ex*scaleWidth)
                    ey = int(ey*scaleHeight)
                    ew = int(ew*scaleWidth)
                    eh = int(eh*scaleHeight)
                    #cv2.rectangle(image,(ex+x,ey+y),(ex+ew+x,ey+eh+y),(0,255,0),2)
                    eyeGray = gray[ey+y:ey+eh+y, ex+x:ex+ew+x]
                    eyesRoi = image[ey+y:ey+eh+y, ex+x:ex+ew+x]
                    cv2.imwrite(filename+str(i)+'.png', eyeGray)

                    hist = cv2.equalizeHist(eyeGray)
                    # cv2.imshow('Histogram',hist)

                    ret,th1 = cv2.threshold(hist,70,255,cv2.THRESH_BINARY)
                    # cv2.imshow('threshold',th1)

                    #Applica l'operatosre Canny all'immagine grigia e la visualizza nella finestra 'Canny'
                    #th1 = cv2.Canny(th1, 5, 70, 3)
                    th1 = cv2.Canny(th1, 5, 70)
                    # cv2.imshow('Canny',th1)

                    # houghCircles = cv2.HoughCircles(th1, cv.CV_HOUGH_GRADIENT, 2, 400.0, param1=150, param2=25, minRadius=40, maxRadius=60)
                    houghCircles = cv2.HoughCircles(th1, cv.CV_HOUGH_GRADIENT, 2, 400.0)

                    if houghCircles is not None:
                        j = 0
                        for k in houghCircles[0,:]:
                            center = (k[0],k[1])
                            radius = k[2]
                            cv2.circle(eyesRoi,center,radius,(0,255,0),2)
                            if args.pupil == '1':
                            	pupilRadius = getPupil(eyeGray, center[0], center[1], radius)
                            elif args.pupil == '2':
                            	pupilRadius = getPupil2(eyeGray, center[0], center[1], radius)
                            else:
                            	pupilRadius = getPupil(eyeGray, center[0], center[1], radius)
                            if pupilRadius is not None:
                                cv2.circle(eyesRoi,center,pupilRadius,(0,255,0),1)
                            cv2.imwrite(filename+str(i)+str(j)+'.png', eyesRoi)
                            # cv2.circle(eyesRoi,center,pupilRadius,(255,255,255),-1)
                            cv2.imwrite(filename+'Pupil'+str(i)+str(j)+'.png', eyesRoi)
                            j = j+1
                    i = i+1


            cv2.imwrite(filename+'.png', imageCopy)
        #cv2.imwrite('/home/pi/Desktop/cv-videoDectFoto.png',frame)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()