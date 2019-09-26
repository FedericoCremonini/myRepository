import cv2
import numpy as np
from utils import *
import os

left = cv2.imread('image_02' + os.path.sep + 'data' + os.path.sep + '0%09d.jpg'%(0),-1)

for i in range(100):
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


  collage0 = np.concatenate((left,disp0),1)
  collage1 = np.concatenate((flow_color.astype(np.uint8),disp1),1)
  collage = np.concatenate((collage0,collage1),0)
  collage = cv2.resize(collage, (left.shape[1], left.shape[0]))

  cv2.imshow('sceneflow', collage)
  cv2.waitKey(1)

  
