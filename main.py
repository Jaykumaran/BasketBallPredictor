import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

#initialize the video
cap = cv2.VideoCapture('Files/Videos/vid (5).mp4')

#find color of object
myColorFinder = ColorFinder(False) #to run in debug to find color True...after finding hsv vals set to false
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}


#Center Position of ball list
posListX=[]
posListY=[]
xList =[item for item in range(0,1300)] #1300 width of img ,this will give [0,1,...,1300]
prediction = False

#********************************************************************************************************************

while True:
    success,img = cap.read()
    # img = cv2.imread("Files/Ball.png")
    img = img[0:900, :]   #cropping the height as low half has some similar color to ball...to avoid miscalc we crop...as img size is  1300* 1080


    #Find the ball color
    imgColor,mask = myColorFinder.update(img,hsvVals)

    #Find Ball location
    imgContours,contours = cvzone.findContours(img,mask,minArea=500)

    if contours:
        posListX.append(contours[0]['center'][0])  # updating center of ball in  all frames for tracking
        posListY.append(contours[0]['center'][1])
        # cx,cy = contours[0]['center']   #findContours the biggest one is already sorted to 0th index...we take only the biggest contour
        # print(cx,cy)
        if posListX:
            # POLYNOMIAL REGRESSSION y =Ax^2 + Bx + C
            #Find coefficeints of poly
            A,B,C = np.polyfit(posListX,posListY,2)  #for quadratic deg=2 and we rely on posList for pts


            for i,(posX,posY) in enumerate(zip(posListX,posListY)):
                pos = (posX,posY)
                cv2.circle(imgContours,pos,10,(0,255,0),cv2.FILLED)
                if i==0:
                    cv2.line(imgContours, pos, pos, (0, 210, 0), 2)
                else:
                    cv2.line(imgContours,pos,(posListX[i-1],posListY[i-1]),(0,210,0),5)

            for x in xList:
                y = int(A * x ** 2 + B * x + C)
                cv2.circle(imgContours,(x,y),3,(0,0,210),cv2.FILLED)

    ###################################################################################################################
            #PREDICTIONS
            #finding the hoop width at a height in the total img dimensions
            # X values in our img for hoop range is 330 to 430 and hoop is at height 590 from origin
            #we put y to find x

            if len(posListX) < 10 : #10 frames bcuz within that baall reaches hoop

                a = A
                b = B
                c = C - 590

                x = int((-b - math.sqrt(b**2 - (4*a*c)))//(2*a))
                prediction = 330 < x < 430  # Predict only upto the ball reaches hoop

            if prediction:            #prediction turns True
                print("Will be inside Basket / Hoop")
                cvzone.putTextRect(imgContours,"Inside Basket",(50,150),scale=5,
                                   thickness=5,colorR=(0,210,0),offset=20)
            else:
                print("Outside Basket")
                cvzone.putTextRect(imgContours, "Outside Basket", (50, 150), scale=5,
                                   thickness=5, colorR=(0, 0, 210),offset=20)


#--------------------**************************************------------------------------------*************************

    #Display
    # img = cv2.resize(img,(0,0),None,0.7,0.7)   #0.7 is scale
    # imgMask = cv2.resize(mask, (0, 0), None, 0.7, 0.7)
    img = cv2.resize(imgColor, (0, 0), None, 0.7, 0.7)
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image",img)
    # cv2.imshow("ImageColor",imgColor)
    cv2.imshow("ImageContours", imgContours)
    # cv2.imshow("Masked", imgMask)
    cv2.waitKey(100)
