#código para verificação de rosto e contagem de pessoas (entrada em ônibus)

import cv2 as cv
import numpy as np
import logging

def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv.VideoCapture('2.mp4') #Verificando rosto e contabilizador de rosto 

bg = cv.createBackgroundSubtractorMOG2()

detects = []

post = 200 #altura das linhas de detecção 
post2 = 606
offset = 5

xy1 = (495, post2)#480
xy2 = (495, post)#495

y1 = (550, 500)#540,500
y2 = (550, 1)#550,1

yy=(433,1)#yy=(433,1)
yy2 = (433, 400)#yy2 = (417, 400)

post5 = 450
offset5 = 300

xy5 = (100, post5)
xy6 = (600, post5)

total = 0

up = 0
down = 0

while 1:

    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    mask = bg.apply(gray)

    retval, th = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations = 2)

    dilation = cv.dilate(opening, kernel, iterations = 8)

    closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel, iterations = 8)

    contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    i = 0
    for c in contours:
        (x,y,w,h) = cv.boundingRect(c)
        area = cv.contourArea(c)

        if int(area) > 3000:
            centro = center(x, y, w, h)

            cv.putText(frame, str(i), (x+5, y+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)                
            cv.circle(frame, centro, 4, (0, 0, 255), -1)
            cv.rectangle(frame, (x,y), (x+w , y+h), (0,255,0), 2)

            if len(detects) <= i:
                detects.append([])
            if centro [1] > post2-offset and centro[1] < post2+offset:
                detects[i].append(centro)
            else:
                detects[i].clear()
            i += 1  

    if i == 0:
        detects.clear()
    
    i = 0

    if len(contours) == 0:
        detects.clear()
    else:

        for detect in detects:
            for(c, l) in enumerate(detect):

                if detect[c-1][1] < post2 and l[1] > post:
                    detect.clear()
                    up += 1
                    total += 1
                    cv.line(frame, xy1, xy2, (0,255,0), 5)
                    logging.basicConfig(filename="log.txt", level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p') #Gerando log quando detectado movimentação de pessoa 
                    logging.info("PESSOAS NO AMBIENTE: "+str(up-down)+"/50")
                    continue

                if detect[c-1][1] > post2 and l[1] < post:
                    detect.clear()
                    down += 1
                    total += 1
                    cv.line(frame, xy1, xy2, (0,0,255), 5)
                    logging.basicConfig(filename="log.txt", level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p') #Gerando log quando detectado movimentação de pessoa 
                    logging.info("PESSOAS NO AMBIENTE: "+str(up-down)+"/50")                    
                    continue
                
                if c > 0:
                    cv.line(frame, detect[c-1], l, (0,0,255), 1)
    
    class_body = cv.CascadeClassifier('haarcascadefrontalface.xml')
    if cap.isOpened():
        ret, frame = cap.read()
        frame = cv.resize(frame, None, fx=0.7, fy=0.7, interpolation = cv.INTER_LINEAR)
        grayscale_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        bodies_detect = class_body.detectMultiScale(grayscale_img, 1.2 , 7)#ajuste fino
        for (x,y,w,h) in bodies_detect:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.imshow("frame", frame)


    cv.line(frame, xy1, xy2, (255,0,0), 3)
    cv.line(frame,(yy[0], post-offset), (yy2[0], post2-offset), (255,255,0), 2)
    cv.line(frame,(y1[0], post2+offset), (y2[0], post+offset), (255,255,0), 2)
 
    cv.putText(frame, "MOVIMENTACAO DE PESSOAS: "+str(total), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2) #quantidade de pessoas detectadas perto da linha de verificação
    cv.putText(frame, "ENTRANDO: "+str(up), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
    cv.putText(frame, "SAINDO: "+str(down), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)
    cv.putText(frame,"NO AMBIENTE: "+str(up-down)+"/50 PESSOAS", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
    cv.imshow("frame", frame)

    k = cv.waitKey(20)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
