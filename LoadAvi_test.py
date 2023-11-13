# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mask_and_filter import mask_and_filter


#Enter the path to read the data
path = r'\\hi.doc.ic.ac.uk\groups\ibe\Endomicroscopy\SMART Project Data\virtual slit  line scan _ 660 nm\data\imaging tests\porcine colon tissue tests\test1'
#import sys
#sys.path.append(path)
#Input the vido
inputFile = 'Menngioma.mpg'
cap = cv2.VideoCapture(inputFile)
#Define the first and final image frame number
startImage = 1
endImage = 1000

# Read the first frame
ret, backgroundImage = cap.read()
backgroundImage = cv2.cvtColor(backgroundImage, cv2.COLOR_BGR2GRAY)
#Use the function mask_and_filter
backgroundImage=mask_and_filter(backgroundImage,290,1,10,'r')
#plt.imshow(backgroundImage, cmap='gray')
cv2.imshow('Background Image', backgroundImage)

if endImage == -1:
    endImage = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

nImages = endImage - startImage + 1
print(nImages)
for i in range(2, nImages+1):
    ret, currentImage = cap.read()
    currentImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
    # Use the function mask_and_filter
    currentImage = mask_and_filter(currentImage,100,1,10,'rec')
    #currentImage = currentImage + 159
    cv2.imshow('Current Image', currentImage)
    print(i)
    #cv2.waitKey()
# relesase the vido
cap.release()
cv2.waitKey()