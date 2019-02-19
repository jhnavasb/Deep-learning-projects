import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from collections import deque
from imutils.video import VideoStream

def Filtro (Img):
    ColorLow = (25, 80, 105)
    ColorHigh = (37, 236, 255)
    hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, ColorLow, ColorHigh)
    kernel = np.ones((3,3),np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, None, iterations=3)
    return mask


bufferSize = 64
pts = deque(maxlen=bufferSize)
vs = VideoStream(src=0).start()

NI = 0
while True:
    Img = vs.read()
    Img = cv2.flip( Img, 1 )
    Mask = Filtro (Img)
    
    cv2.imshow("Frame", Mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        cv2.imwrite('4_OtrosE'+format(NI)+'.jpg',Mask)
        NI += 1
        if NI == 50:
            break

vs.stop()
cv2.destroyAllWindows()

