import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time


from collections import deque
from imutils.video import VideoStream

bufferSize = 64
pts = deque(maxlen=bufferSize)
vs = VideoStream(src=0).start()

NI = 150
while True:
    Img = vs.read()
    #Img = cv2.flip( Img, 1 )
    cv2.imshow("Frame", Img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    time.sleep(1)
    cv2.imwrite('3_'+'Puño_Jhonatan'+format(NI)+'.jpg',Img)
    NI += 1
    if NI == 180:
        break 

vs.stop()
cv2.destroyA9llWindows()

#labels = ['Mano abierta', 'Palma', 'Puño', 'Indice']