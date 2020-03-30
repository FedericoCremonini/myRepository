import numpy as np
from utils import *
import os
import random
import math

def find_traslation_z(disp0, disp1, x, y, focale, baseline, min, max):

    """
    :param disp0: disparity map 0
    :param disp1: disparity map 1
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :param baseline: baseline
    :param min: bottom bound traslation
    :param max: upper bound traslation
    :return mTz: estimated traslation along z
    """
    
    depth0 = focale*baseline/disp0
    depth1 = focale*baseline/disp1

    depth2 = depth0-depth1
    depth2 = np.reshape(depth2, (x*y, 1))

#standard solution

    depth0 = np.reshape(depth0, (x*y, 1))
    depth2 = [depth2[i] for i in range(len(depth2)) if (depth0[i] < max and depth0[i] > min and depth2[i] > 0)]

    mTz = np.average(depth2)

#alternative more efficient
    
    #depth2 = np.clip(depth2, min, max)
    #mTz = np.median(depth2)
    
    return mTz

def find_traslation_z(depth0, depth1, x, y, min, max):

    """
    :param depth0: depth map 0
    :param depth1: depth map 1
    :param x: width of the maps
    :param y: height of the maps
    :param min: bottom bound traslation
    :param max: upper bound traslation
    :return mTz: estimated traslation along z
    """

    depth2 = depth0-depth1

    depth2 = np.reshape(depth2, (x*y, 1))

#standard solution

    depth0 = np.reshape(depth0, (x*y, 1))
    depth2 = [depth2[i] for i in range(len(depth2)) if (depth0[i] < max and depth0[i] > min and depth2[i] > 0)]

    mTz = np.average(depth2)

#alternative more efficient
    
    #depth2 = np.clip(depth2, min, max)
    #mTz = np.median(depth2)
    
    return mTz

def flow_correction_tras_z(flow, disp0, traslation_z, x, y, focale, baseline):

    """
    :param flow: original flow map
    :param disp0: disparity map 0
    :param disp1: disparity map 1
    :param traslation_z: estimated traslation along z
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :param baseline: baseline
    :return flow_correction: correction flow map
    :return flow_corrected: corrected flow map
    """

    v = []
    r = []
    
    for i in range(y):
        for j in range(x):

            depth0 = (focale * baseline) / disp0[i][j] 
            depth1 = depth0 - traslation_z 

            x0 = int(j - x/2) ; x1 = (depth0/depth1)*x0 
            y0 = int(y/2 - i) ; y1 = (depth0/depth1)*y0

            flow_x = (x1 - x0)
            flow_y = -1.0*(y1 - y0)

            v.append([flow_x, flow_y])
            r.append([(flow[i][j][0]-flow_x), (flow[i][j][1]-flow_y)])

    flow_correction = np.array(v)
    flow_correction = np.reshape(flow_correction, (y, x, 2))

    flow_corrected = np.array(r)
    flow_corrected = np.reshape(flow_corrected, (y, x, 2))

    return flow_correction, flow_corrected

def flow_correction_tras_z(flow, depth0, traslation_z, x, y):

    """
    :param flow: original flow map
    :param depth0: depth map 0
    :param depth1: depth map 1
    :param traslation_z: estimated traslation along z
    :param x: width of the maps
    :param y: height of the maps
    :return flow_correction: correction flow map
    :return flow_corrected: corrected flow map
    """

    v = []
    r = []

    for i in range(y):
        for j in range(x):

            depth1 = depth0 - traslation_z 

            x0 = int(j - x/2) ; x1 = (depth0/depth1)*x0 
            y0 = int(y/2 - i) ; y1 = (depth0/depth1)*y0

            flow_x = (x1 - x0)
            flow_y = -1.0*(y1 - y0)

            v.append([flow_x, flow_y])
            r.append([(flow[i][j][0]-flow_x), (flow[i][j][1]-flow_y)])

    flow_correction = np.array(v)
    flow_correction = np.reshape(flow_correction, (y, x, 2))

    flow_corrected = np.array(r)
    flow_corrected = np.reshape(flow_corrected, (y, x, 2))

    return flow_correction, flow_corrected

def find_traslation_xy(flow, disp0, x, y, focale, baseline):

    """
    :param flow: original flow map
    :param disp0: disparity map 0
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :param baseline: baseline
    :return mTx: estimated traslation along x
    :return mTy: estimated traslation along y
    """
    
    tx =[]
    ty = []

    for i in range(y):
         for j in range(x):

             depth0 = (focale * baseline) / disp0[i][j]

             t_x = (depth0/focale) * flow[i][j][0]
             tx.append(t_x)
             
             t_y = (depth0/focale) * flow[i][j][1]
             ty.append(t_y)

    mTx = np.median(tx)
    mTy = np.median(ty)

    return mTx, mTy

def find_traslation_xy(flow, disp0, x, y, focale, baseline, minX, maxX, minY, maxY):

    """
    :param flow: original flow map
    :param disp0: disparity map 0
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :param baseline: baseline
    :param minX: bottom bound traslation along x
    :param maxX: upper bound traslation along x
    :param minY: bottom bound traslation along y
    :param maxY: upper bound traslation along y
    :return mTx: estimated traslation along x
    :return mTy: estimated traslation along y
    """
    
    tx =[]
    ty = []

    for i in range(y):
         for j in range(x):

             depth0 = (focale * baseline) / disp0[i][j]

             t_x = (depth0/focale) * flow[i][j][0]
             if(t_x > minX and t_x < maxX):
                tx.append(t_x)

             t_y = (depth0/focale) * flow[i][j][1]
             if(t_y > minY and t_y < maxY):
                ty.append(t_y)

    mTx = np.median(tx)
    mTy = np.median(ty)

    return mTx, mTy

def find_traslation_xy(flow, depth0, x, y, focale):

    """
    :param flow: original flow map
    :param depth0: depth map 0
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :return mTx: estimated traslation along x
    :return mTy: estimated traslation along y
    """
    
    tx =[]
    ty = []

    for i in range(y):
         for j in range(x):

             t_x = (depth0/focale) * flow[i][j][0]
             tx.append(t_x)

             t_y = (depth0/focale) * flow[i][j][1]
             ty.append(t_y)

    mTx = np.median(tx)
    mTy = np.median(ty)

    return mTx, mTy

def find_traslation_xy(flow, depth0, x, y, focale, minX, maxX, minY, maxY):

    """
    :param flow: original flow map
    :param depth0: dwpth map 0
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :param minX: bottom bound traslation along x
    :param maxX: upper bound traslation along x
    :param minY: bottom bound traslation along y
    :param maxY: upper bound traslation along y
    :return mTx: estimated traslation along x
    :return mTy: estimated traslation along y
    """
    
    tx =[]
    ty = []

    for i in range(y):
         for j in range(x):

             t_x = (depth0/focale) * flow[i][j][0]
             if(t_x > minX and t_x < maxX):
                tx.append(t_x)

             t_y = (depth0/focale) * flow[i][j][1]
             if(t_y > minY and t_y < maxY):
                ty.append(t_y)

    mTx = np.median(tx)
    mTy = np.median(ty)

    return mTx, mTy

def flow_correction_tras_xy(flow, disp0, traslation_x, traslation_y, x, y, focale, baseline):

    """
    :param flow: original flow map
    :param disp0: disparity map 0
    :param traslation_x: estimated traslation along x
    :param traslation_y: estimated traslation along y
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :param baseline: baseline
    :return flow_correction: correction flow map
    :return flow_corrected: corrected flow map
    """

    v = []
    r = []

    for i in range(y):
         for j in range(x):

             depth0 = (focale * baseline) / disp0[i][j]

             flow_x = (focale/depth0) * traslation_x
             flow_y = (focale/depth0) * traslation_y

             v.append([flow_x, flow_y])
             r.append([(flow[i][j][0]-flow_x), (flow[i][j][1]-flow_y)])

    flow_correction = np.array(v)
    flow_correction = np.reshape(flow_correction, (y, x, 2))

    flow_corrected = np.array(r)
    flow_corrected = np.reshape(flow_corrected, (y, x, 2))

    return flow_correction, flow_corrected

def flow_correction_tras_xy(flow, depth0, traslation_x, traslation_y, x, y, focale):

    """
    :param flow: original flow map
    :param depth0: depth map 0
    :param traslation_x: estimated traslation along x
    :param traslation_y: estimated traslation along y
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :return flow_correction: correction flow map
    :return flow_corrected: corrected flow map
    """

    v = []
    r = []

    for i in range(y):
         for j in range(x):

             flow_x = (focale/depth0) * traslation_x
             flow_y = (focale/depth0) * traslation_y

             v.append([flow_x, flow_y])
             r.append([(flow[i][j][0]-flow_x), (flow[i][j][1]-flow_y)])

    flow_correction = np.array(v)
    flow_correction = np.reshape(flow_correction, (y, x, 2))

    flow_corrected = np.array(r)
    flow_corrected = np.reshape(flow_corrected, (y, x, 2))

    return flow_correction, flow_corrected

def find_rotation_xy(flow, x, y, focale):

    """
    :param flow: original flow map
    :param x: width of the map
    :param y: height of the map
    :param focale: focale
    :return mRx: estimated horizontal rotation
    :return mRy: estimated vertical rotation
    """
    
    rx =[]
    ry = []

    for i in range(y):
         for j in range(x):

             x0 = int(j - x/2) ; x1 = x0 + flow[i][j][0]  
             
             r_x = (focale * (x0 - x1))/(x0*x1 + focale*focale)
             rx.append(r_x)

             y0 = int(y/2 - i) ; y1 = y0 - flow[i][j][1] 
             
             r_y = (focale * (y0 - y1))/(y0*y1 + focale*focale)
             ry.append(r_y)

    mRx = np.median(rx)
    mRy = np.median(ry)

    return mRx, mRy

def find_rotation_xy(flow, x, y, focale,  minX, maxX, minY, maxY):

    """
    :param flow: original flow map
    :param x: width of the map
    :param y: height of the map
    :param focale: focale
    :param minX: bottom bound horizontal rotation
    :param maxX: upper bound horizontal rotation
    :param minY: bottom bound vertical rotation
    :param maxY: upper bound vertical rotation
    :return mRx: estimated horizontal rotation
    :return mRy: estimated vertical rotation
    """
    
    rx =[]
    ry = []

    for i in range(y):
         for j in range(x):

             x0 = int(j - x/2) ; x1 = x0 + flow[i][j][0] 
             
             r_x = (focale * (x0 - x1))/(x0*x1 + focale*focale)
             if(r_x > minX and r_x < maxX):
                rx.append(r_x)

             y0 = int(y/2 - i) ; y1 = y0 - flow[i][j][1] 
             
             r_y = (focale * (y0 - y1))/(y0*y1 + focale*focale)
             if(r_y > minY and r_y < maxY):
                ry.append(r_y)

    mRx = np.median(rx)
    mRy = np.median(ry)

    return mRx, mRy

def flow_correction_tras_xy(flow, rotation_x, rotation_y, x, y, focale):

    """
    :param flow: original flow map
    :param rotation_x: estimated horizontal rotation
    :param rotation_y: estimated vertical rotation
    :param x: width of the maps
    :param y: height of the maps
    :param focale: focale
    :return flow_correction: correction flow map
    :return flow_corrected: corrected flow map
    """

    v = []
    r = []

    for i in range(y):
         for j in range(x):

             x0 = int(j - x/2)
             x1 = (focale * (x0 - focale * rotation_x))/(focale + x * rotation_x)
             flow_x = x1 - x0

             y0 = int(y/2 - i)
             y1 = (focale * (y0 - focale * rotation_y))/(focale + y * rotation_y)
             flow_y = -1.0*(y1 - y0)

             v.append([flow_x, flow_y])
             r.append([(flow[i][j][0]-flow_x), (flow[i][j][1]-flow_y)])

    flow_correction = np.array(v)
    flow_correction = np.reshape(flow_correction, (y, x, 2))

    flow_corrected = np.array(r)
    flow_corrected = np.reshape(flow_corrected, (y, x, 2))

    return flow_correction, flow_corrected

def flow_filter(flow, bound):

    """
    :param flow: original flow map
    :param bound: bound flow
    :return mask: mask
    """

    norm = np.linalg.norm(flow, axis = -1)
    mask = norm > bound
    mask = np.stack((mask, mask), axis = -1)

    return mask



