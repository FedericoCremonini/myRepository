import cv2
import numpy as np
from utils import *
import os
import random
import math

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, ch = img1.shape
    # Da utilizzare se utilizzato cv2.imread(img, 0)
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def foundIntersection(lines):
    ''' lines- vettore di epipolar lines'''
    
    r1 = lines[int(random.randrange(0, lines1.size/3))]; a1 = r1[0] ; b1 = r1[1] ; c1 = r1[2] # a1 * x + b1 * y + c1 = 0
    r2 = lines[int(random.randrange(0, lines1.size/3))]; a2 = r2[0] ; b2 = r2[1] ; c2 = r2[2] # a2 * x + b2 * y + c2 = 0
    x = (b1/b2*c2 - c1)/(a1-b1/b2*a1)
    y = (a2/a1*c1 - c2)/(b2-a2/a1*b1)
        
    if math.isfinite(x) :
        x = int(x)
    else:
        x = -1

    if math.isfinite(y) :
        y = int(y)
    else:
        y = -1


    return x, y

def foundEpipole(lines1, lines2, n):
    '''n numero di iterazioni utilizzate per trovare intersezione '''

    v1x = []
    v1y = []

    v2x = []
    v2y = []

    for i in range(n):
        x1, y1 = foundIntersection(lines1);
        x2, y2 = foundIntersection(lines2);
        if x1 >= 0 and x1 < 1241 :
            v1x.append(x1)
        if y1 >= 0  and y1 < 376 :
            v1y.append(y1)
        if x2 >= 0 and x2 < 1241 :
            v2x.append(x2)
        if y2 >= 0  and y2 < 376 :
            v2y.append(y2)

    v1x.sort()
    v1y.sort()
    v2x.sort()
    v2y.sort()

    return v1x, v1y, v2x, v2y



for i in range(100):
  if(i == 0):
      old = cv2.imread('image_02' + os.path.sep + 'data' + os.path.sep + '0%09d.jpg'%(0),-1)
  else :
      old = cv2.imread('image_02' + os.path.sep + 'data' + os.path.sep + '0%09d.jpg'%(i-1),-1)

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
  
  keypointsOld, descriptorsOld = orb.detectAndCompute(old, None)
  descriptorsOld = np.float32(descriptorsOld)
  kpold = cv2.drawKeypoints(old, keypointsOld, None)

  keypointsLeft, descriptorsLeft = orb.detectAndCompute(left, None)
  descriptorsLeft = np.float32(descriptorsLeft)
  kpleft = cv2.drawKeypoints(left, keypointsLeft, None)
  
  # FLANN parameters ######################################################################
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params,search_params)
  matches = flann.knnMatch(descriptorsLeft,descriptorsOld,k=2)

  good = []
  pts1 = []
  pts2 = []
  # ratio test as per Lowe's paper
  for i,(m,n) in enumerate(matches):
    if m.distance < 0.85*n.distance:
        good.append(m)
        pts2.append(keypointsOld[m.trainIdx].pt)
        pts1.append(keypointsLeft[m.queryIdx].pt)
  # We find fundmental matrix
  pts1 = np.int32(pts1)
  pts2 = np.int32(pts2)
  F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
  # We select only inlier points
  pts1 = pts1[mask.ravel()==1]
  pts2 = pts2[mask.ravel()==1]
  # Find epilines corresponding to points in right image (second image) and
  # drawing its lines on left image
  lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
  lines1 = lines1.reshape(-1,3)
  img5, img6 = drawlines(left,old,lines1,pts1,pts2)
  # Find epilines corresponding to points in left image (first image) and
  # drawing its lines on right image
  lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
  lines2 = lines2.reshape(-1,3)
  img3, img4 = drawlines(old,left,lines2,pts2,pts1)
 
  #find epipole
  
  #r11 = lines1[int(random.randrange(0, lines1.size/3))]; a11 = r11[0] ; b11 = r11[1] ; c1 = r11[2] # a11 * x + b11 * y + c1 = 0
  #r12 = lines1[int(random.randrange(0, lines1.size/3))]; a2 = r12[0] ; b2 = r12[1] ; c2 = r12[2] # a2 * x + b2 * y + c2 = 0



  #r11 = lines1[50]; a11 = r11[0] ; b11 = r11[1] ; c1 = r11[2] # a11 * x + b11 * y + c1 = 0
  #r12 = lines1[200]; a2 = r12[0] ; b2 = r12[1] ; c2 = r12[2] # a2 * x + b2 * y + c2 = 0

  #x = round((b1/b2*c2 - c1)/(a11-b1/b2*a11))
  
  #y = round((a2/a1*c1 - c2)/(b2-a2/a1*b1))
  
  v1x, v1y, v2x, v2y = foundEpipole(lines1, lines2, 100)


  print("x1= " + str(v1x))
  print("y1= " + str(v1y))

  print("x2= " + str(v2x))
  print("y2= " + str(v2y) + "\n")
  
  
  #if(x > 0 and y > 0) :
    #for j in range(0, 0):#y
      #for k in range (0, 0):#x
          #left[j][k] = [0, 0, 0]



  collage0 = np.concatenate((left,disp0, kpold, img3),1)
  collage1 = np.concatenate((flow_color.astype(np.uint8),disp1, kpleft, img5),1)
  collage = np.concatenate((collage0,collage1),0)
  collage = cv2.resize(collage, (left.shape[1], left.shape[0]))

  cv2.imshow('sceneflow', collage)
  
  #cv2.waitKey(0) #per stoppare ogni frame
  cv2.waitKey(1) # per vedere video
  