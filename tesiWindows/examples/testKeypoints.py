import cv2
import numpy as np
from utils import *
import os

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

left = cv2.imread('image_02' + os.path.sep + 'data' + os.path.sep + '0%09d.jpg'%(0),-1)
keypointsLeft = None
descriptorsLeft = None
kpleft = None

for i in range(100):
  old = left
  left = cv2.imread('image_02' + os.path.sep + 'data' + os.path.sep + '0%09d.jpg'%i,-1)

  # Per aprire disp0 e disp1: dividi per 256
  disp0 = (cv2.imread('data' + os.path.sep + 'disp_0' + os.path.sep + '%06d_10.png'%i,-1)/256).astype(np.uint8)
  disp1 = (cv2.imread('data' + os.path.sep + 'disp_1' + os.path.sep + '%06d_10.png'%i,-1)/256).astype(np.uint8)
  # da disp0, si ottiene la depth facendo focale*baseline/disp0, focale=720 baseline=0.54

  # colormap per visualizzazione
  disp0 = cv2.applyColorMap(disp0, 2)
  disp1 = cv2.applyColorMap(disp1, 2)

  # Per aprire flow: converti in float, sottrai 2**15 e dividi per 64
  flow = cv2.imread('data' + os.path.sep + 'flow' + os.path.sep + '%06d_10.png'%i,-1).astype(np.float32)
  flow = (flow-(2**15))/64.

  # colormap per visualizzazione
  flow = np.concatenate((flow[:,:,2:3],flow[:,:,1:2]),-1)
  flow_color = cv2.cvtColor(flow_to_image(flow), cv2.COLOR_BGR2RGB)

  # Per ricerca epipole -----------------------------------------------------------------------------
  
  #sift = cv2.xfeatures2d.SIFT_create()
  orb = cv2.ORB_create(nfeatures=1500)

  #per trovare epipole 
  if i == 0 :
      keypointsOld, descriptorsOld = orb.detectAndCompute(old, None)
      kpold = cv2.drawKeypoints(old, keypointsOld, None)
  else :
    keypointsOld = keypointsLeft
    descriptorsOld = descriptorsLeft
    kpold = kpleft

  keypointsLeft, descriptorsLeft = orb.detectAndCompute(left, None)
  kpleft = cv2.drawKeypoints(left, keypointsLeft, None)
  print(len(keypointsOld))
      

  
  cv2.waitKey(1)
