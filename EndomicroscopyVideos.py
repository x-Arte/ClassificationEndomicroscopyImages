import torch
import numpy as np
import pandas as pd
import cv2
from mask_and_filter import mask_and_filter
import os
from EndomicroscopyImage import EndomicroscopyImage

def load_dic(dicname,radius_value,hsize_value,sigma_value,outputpath,startImage = 1,endImage = 10):
    if not(dicname):
        raise NotADirectoryError
    cnt = 0
    for filename in os.listdir(dicname):
        print(filename)
        cnt+=1
        label = -1 #-1:none 0:GBM 1:meningioma etc.
        if filename.startswith('GBM'):
            label = 1
        elif filename.startswith('meningioma'):
            label = 2
        cap = cv2.VideoCapture(dicname+filename)
        # Define the first and final image frame number

        ret, backgroundImage = cap.read()
        backgroundImage = cv2.cvtColor(backgroundImage, cv2.COLOR_BGR2GRAY)
        # Use the function mask_and_filter
        backgroundImage = mask_and_filter(backgroundImage, radius_value, hsize_value, sigma_value, 'rec')
        # plt.imshow(backgroundImage, cmap='gray')
        cv2.imshow('Background Image', backgroundImage)
        if endImage == -1:
            endImage = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        nImages = endImage - startImage + 1
        #print(nImages)
        for i in range(2, nImages + 1):
            ret, currentImage = cap.read()
            currentImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
            # Use the function mask_and_filter
            currentImage = mask_and_filter(currentImage, radius_value,hsize_value,sigma_value, 'rec')
            # currentImage = currentImage + 159
            cv2.imshow('Current Image', currentImage)
            eimg = EndomicroscopyImage(filename[:filename.rfind('.')],i,label,currentImage)
            eimg.save(outputpath)
            print(i)
            # cv2.waitKey()
        # relesase the vido
        cap.release()

    return cnt

if __name__ == "__main__":
    dicname = "dataset/meningioma/"
    radius_value = 150
    hsize_value = 1
    sigma_value = 1.2
    outputpath = 'dataset/images/'
    cnt = load_dic(dicname,radius_value,hsize_value,sigma_value,outputpath)
    print(cnt)

