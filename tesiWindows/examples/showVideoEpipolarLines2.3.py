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

def bestPoint(res, x, y):
    best = 0
    bestX = -1
    bestY = -1

    for i in range(x):
        for j in range(y): 
            if( res[i][j] > best):
                bestX = i
                bestY = j
                best = res[i][j]
     
    return bestX, bestY, best

def findEpipole2(lines, n, x, y):

    best = 0.0
    bestX = -1
    bestY = -1

    res = [ [ 0 for i in range(y) ] for j in range(x) ]
    #res = np.zeros((x,y))
    for i in range(n):
        r = lines[int(random.randrange(0, lines1.size/3))]; a = r[0] ; b = r[1] ; c = r[2] # a * x + b * y + c = 0
        for j in range(x):
            coordY = int(-(a * j  + c)/b)

            if(coordY >= 0 and coordY < y):
                res[j][coordY] = res[j][coordY]+1
                if(res[j][coordY] == 5):
                    bestX = j
                    bestY = coordY
                    best = 1.0
            if(best == 1.0):
                break
        if(best == 1.0):
                break

    if(best != 1.0):
        bestX, bestY, best = bestPoint(res, x, y)
        best = best/n

    return bestX, bestY, best

x = 1241 #larghezza in pixel immagini in input
y = 376 #altezza in pixel immagini in input
focale = 720
baseline = 0.54
precTot = 0

flowCorrect = [ [ [0 for k in range(3) ]for i in range(y) ] for j in range(x) ]

for it in range(0,100):
  if(it == 0):
      old = cv2.imread('image_02' + os.path.sep + 'data' + os.path.sep + '0%09d.jpg'%(0),-1)
  else :
      old = cv2.imread('image_02' + os.path.sep + 'data' + os.path.sep + '0%09d.jpg'%(it-1),-1)

  left = cv2.imread('image_02' + os.path.sep + 'data' + os.path.sep + '0%09d.jpg'%it,-1)
  # Per aprire disp0 e disp1: dividi per 256
  disp0 = (cv2.imread('data' + os.path.sep + 'disp_0' + os.path.sep + '%06d_10.png'%it,-1)/256).astype(np.uint8)
  disp1 = (cv2.imread('data' + os.path.sep + 'disp_1' + os.path.sep + '%06d_10.png'%it,-1)/256).astype(np.uint8)
  # da disp0, si ottiene la depth facendo focale*baseline/disp0, focale=720 baseline=0.54

  # colormap per visualizzazione
  disp0vis = cv2.applyColorMap(disp0, 2)
  disp1vis = cv2.applyColorMap(disp1, 2)

  # Per aprire flow: converti in float, sottrai 2**15 e dividi per 64
  flow = cv2.imread('data' + os.path.sep + 'flow' + os.path.sep + '%06d_10.png'%it,-1).astype(np.float32)
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

  coordX, coordY, prec = findEpipole2(lines1, 10, x, y)

  bEp = 255 - flow_color.astype(np.uint8)[coordY][coordX][0]
  gEp = 255 - flow_color.astype(np.uint8)[coordY][coordX][1]
  rEp = 255 - flow_color.astype(np.uint8)[coordY][coordX][2]
  depth0 = (focale * baseline) / disp0[coordY][coordX]
  depth1 = (focale * baseline) / disp1[coordY][coordX]


  res = []
  for i in range(1000):
      p = keypointsLeft[int(random.randrange(0, 1500))].pt ; x = int(p[0]) ; y = int(p[1])
      depth0 = (focale * baseline) / disp0[y][x]
      depth1 = (focale * baseline) / disp1[y][x]
      if(depth0 < 10):
          dif = round((depth0 - depth1), 2)
          res.append(dif)
      
  res.sort()

  differenza = res[int(int(len(res))/2)]

  flowCorrect = flow_color.astype(np.uint8).copy()
  
  #for i in range(y):
      #for j in range(x):
          #flowCorrect[i][j][0] = flowCorrect[i][j][0] + bEp
          #if(flowCorrect[i][j][0] < bEp): flowCorrect[i][j][0] = 255
          #flowCorrect[i][j][1] = flowCorrect[i][j][1] + gEp
          #if(flowCorrect[i][j][1] < gEp): flowCorrect[i][j][1] = 255
          #flowCorrect[i][j][2] = flowCorrect[i][j][2] + rEp
          #if(flowCorrect[i][j][2] < rEp): flowCorrect[i][j][2] = 255


  for i in range(y):
     for j in range(x):
         depth0 = (focale * baseline) / disp0[i][j]
         depth1 = depth0 - differenza
         x0 = int(i - 635) ; x1 = int((depth0/depth1)*x0)
         y0 = int(187 - j) ; y1 = int((depth0/depth1)*y0)
         difX = x1 - x0
         difY = y1 - y0
         if( difX > 0):
             bX = int(pow(3, 1/2)*difX/3)
             gX = int(2 * pow(3, 1/2)*difX/3)
             rX = 0
         if( difX < 0):
             bX = int((-1)* pow(3, 1/2)*difX/3)
             gX = 0
             rX = int((-2) * pow(3, 1/2)*difX/3)
         if( difY > 0):
             bY = 0
             #gY = int((-1)* difY)
             #rY = int((-1)* difY)
             gY = int(difY)
             rY = int(difY)
         if( difY < 0):
             bY = int((-1)* difY)
             #bY = int(difY)
             gY = 0
             rY = 0

         flowCorrect[i][j][0] = flowCorrect[i][j][0] + (bX + bY)
         if(flowCorrect[i][j][0] < (bX + bY)): flowCorrect[i][j][0] = 255
         flowCorrect[i][j][1] = flowCorrect[i][j][1] + (gX + gY)
         if(flowCorrect[i][j][1] < (gX + gY)): flowCorrect[i][j][1] = 255
         flowCorrect[i][j][2] = flowCorrect[i][j][2] + (rX + rY)
         if(flowCorrect[i][j][2] < (rX + rY)): flowCorrect[i][j][2] = 255

  


  collage0 = np.concatenate((left,disp0vis, kpold, img5, flowCorrect),1)
  collage1 = np.concatenate((flow_color.astype(np.uint8),disp1vis, kpleft, img3, flowCorrect),1)
  collage = np.concatenate((collage0,collage1),0)
  collage = cv2.resize(collage, (left.shape[1], left.shape[0]))

  cv2.imshow('sceneflow', collage)
  
  cv2.waitKey(0) #per stoppare ogni frame
  #cv2.waitKey(1) # per vedere video
 
