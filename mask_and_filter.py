import numpy as np
import cv2
import matplotlib.pyplot as plt


def mask_and_filter(previousImage, radius, hsize, sigma, type = 'r'):
    if (type == 'r'):
        imageSize = previousImage.shape
        ci = (imageSize[1] // 2, imageSize[0] // 2, radius)
        # define the mask
        xx, yy = np.meshgrid(np.arange(imageSize[1]) - ci[0], np.arange(imageSize[0]) - ci[1])
        mask = ((xx ** 2 + yy ** 2) < ci[2] ** 2).astype(np.uint8)
        masklogical = mask.astype(bool)

        #mask = np.zeros(imageSize, dtype=np.uint8)
        mask = ((xx ** 2 + yy ** 2) < ci[2] ** 2).astype(np.uint8)
    elif(type == 'rec'):
        imageSize = previousImage.shape
        ci = (imageSize[0] // 2, imageSize[1] // 2, radius)
        print(ci,imageSize)
        # define the mask
        xx, yy = np.meshgrid(np.arange(imageSize[0]) - ci[0], np.arange(imageSize[1]) - ci[1])
        mask = np.zeros(imageSize, dtype=np.uint8)
        mask[ci[0]-radius:ci[0]+radius, ci[1]-radius:ci[1]+radius] = 1
        masklogical = mask.astype(bool)

    #crop the image
    croppedImage = np.zeros_like(masklogical, dtype=previousImage.dtype)
    croppedImage[masklogical] = previousImage[masklogical]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    boundedImage = croppedImage[y:y + h, x:x + w]

    G = cv2.getGaussianKernel(hsize, sigma)
    G = G @ G.T  # Create a 2D Gaussian kernel
    correctedRawImage = cv2.filter2D(boundedImage, -1, G, borderType=cv2.BORDER_REPLICATE)
    return correctedRawImage


if __name__ == '__main__':

    # Read the input image
    input_image = cv2.imread("Queen's_Tower.jpg", cv2.IMREAD_GRAYSCALE)

    # define the radius, hsize, sigma
    radius_value = 200
    hsize_value = 5
    sigma_value = 10

    # call the function
    corrected_image = mask_and_filter(input_image, radius_value, hsize_value, sigma_value,'rec')

    # original image and the Corrected image
    plt.imshow(input_image, cmap='gray')
    plt.title("Original Image")
    plt.show()
    plt.imshow(corrected_image, cmap='gray')
    plt.title("Corrected Image")
    plt.title("Corrected Image")
    plt.show()
