##
## File:	OpenCV_ImageProcessing_Utilities.py
## Author:	Nicolas ROUGON
## Affiliation:	Institut Polytechnique de Paris | Telecom SudParis | ARTEMIS Department
## Date:	July 28, 2022
##
## Description:	OpenCV sample routines
## > Miscellaneous image processing routines
##   - Compute contrast map using Sobel gradient filter
##   - Difference of Gaussian (DoG) Laplacian filter
##   - Find zero crossings of a float (single channel) 2D array
##

import cv2 as cv
import numpy as np
import math
from enum import Enum

class SecondOrderMoment(Enum):
    STD_DEVIATION = 1
    VARIANCE = 2
#
# Compute local variance map: sigma^2(I) = E(I^2) - (E(I))^2
# - Expectation is implemented using a normalized averaging filter blur()
#   > docs.opencv.org/4.6.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37
#
def LocalVarianceMap(img, ksize, type, out):
    # Convert image to float array
    if img.dtype != np.float32:
        the_img = img.astype(dtype="float32")
    else:
        the_img = img
        
    # Compute image local mean
    img_mean = cv.blur(the_img, (ksize,ksize), None)
    
    # Compute image squared local mean
    img2_mean = cv.blur(np.multiply(the_img,the_img), (ksize,ksize), None)

    out = np.subtract(img2_mean, np.multiply(img_mean,img_mean))
    
    if type is SecondOrderMoment.STD_DEVIATION:
        # np.absolute() is used to safely handle sign exceptions which may result
        # from round-off errors
        out = np.sqrt(np.absolute(out))    
    
    return out

#
# Compute contrast map using Sobel filter
# - Sobel gradient filter
#   > docs.opencv.org/4.6.0/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
#
# Tutorial
# - Sobel operator
#   > docs.opencv.org/4.6.0/d2/d2c/tutorial_sobel_derivatives.html
#
def SobelContrastMap(img, norm, out):
    # Compute Sobel gradient components
    gradx = cv.Sobel(img, cv.CV_32F, 1, 0)
    grady = cv.Sobel(img, cv.CV_32F, 0, 1)
    
    # Compute contrast map using numpy
    if norm == cv.NORM_L1:
        out = np.add(np.absolute(gradx),np.absolute(grady))
    elif norm == cv.NORM_L2:
        out = np.sqrt(np.add(np.multiply(gradx,gradx), np.multiply(grady,grady)))
        
    return out

#
# Difference of Gaussian (DoG) filter
# - Gaussian filtering
#   > docs.opencv.org/4.6.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
#
def DoGFilter(img, sigma1, sigma2, method, out):
    # The image returned by GaussianBlur() has the same type as the input image
    # For higher accuracy, convert input image to float array if appropriate
    if img.dtype != np.float32:
        the_img = img.astype(dtype="float32")
    else:
        the_img = img

    # Gaussian kernel size computation
    if method == 0:     # Kernel size not specified
        # cv.GaussianBlur() computes kernel size as:
        #   ksize = cvRound(sigma*(img.depth == CV_8U ? 3 : 4)*2 + 1) | 1
        # Source code: $(OPENCVDIR)/sources/modules/imgproc/src/smooth.cpp
        
        if the_img.dtype == np.float32:
            scaling = 4
        else:
            scaling = 3
        scaling *= 2
        gaussian_ksize1 = round(scaling*sigma1* + 1) | 1
        gaussian_ksize2 = round(scaling*sigma2* + 1) | 1

        kernel_size1 = (0,0)
        kernel_size2 = (0,0)
    elif method == 1:   # Optimal kernel size is computed, yielding narrower kernels
        # Optimal Gaussian kernel size is given by: 
        #            sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
        # i.e.       ksize = 2*((sigma-0.8)/0.3) + 3
        # > docs.opencv.org/4.6.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa

        gaussian_ksize1 = (math.ceil((sigma1 - 0.8)/0.3))*2 + 3
        gaussian_ksize2 = (math.ceil((sigma2 - 0.8)/0.3))*2 + 3        
        kernel_size1 = (gaussian_ksize1, gaussian_ksize1)
        kernel_size2 = (gaussian_ksize2, gaussian_ksize2)
    
    # Gaussian filtering
    out1 = cv.GaussianBlur(the_img, kernel_size1, sigma1)
    out2 = cv.GaussianBlur(the_img, kernel_size2, sigma2)

    # Subtract filter outputs
    out = np.subtract(out1, out2)

    return out, gaussian_ksize1, gaussian_ksize2

#
# Find zero crossings of a float scalar 2D array
# - Fast Python implementation based on mathematical morphology
#   Adapted from: stackoverflow.com/questions/25105916/laplacian-of-gaussian-in-opencv-how-to-find-zero-crossings
#   In the original version, element = np.ones((3,3)) (i.e. 8-connected neighborhood) 
#   > This yields thick edges, due to multiple responses
#
# Documentation
# - Greylevel morphological operators
#   > docs.opencv.org/4.6.0/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
#
def FindZeroCrossings(src, out):
    # Upper-left neighborhood > structuring element
    element = np.array([[1,1,0],[1,1,0],[0,0,0]], dtype="uint8")
    
    # Morphological erosion > local minimum value
    erosion = cv.morphologyEx(src, cv.MORPH_ERODE, element)
    
    # Morphological dilation > local maximum value
    dilation = cv.morphologyEx(src, cv.MORPH_DILATE, element)
    
    # Test for sign change between current pixel and local extremum > zero-crossings
    bool_out = np.logical_or(np.logical_and(erosion < 0,  src > 0), 
                             np.logical_and(dilation > 0, src < 0))
    
    # Convert to 8-bit {0,255} binary image 
    out = 255*bool_out.astype(dtype="uint8")
    
    return out

#
# Find zero crossings of a float scalar 2D array
# - Slower Python implementation based on sign change between current and north / west pixel
#
# Tutorial
# - Image rasterscan
#   > docs.opencv.org/4.6.0/db/da5/tutorial_how_to_scan_images.html
#
def FindZeroCrossings2(src, out):
    threshold = 0.0
    
    nrows, ncols = np.shape(src) 
    for j in range(nrows):
        if j == 0:
            n_index = 1
        else:
            n_index = -1
        p_prev = src[j+n_index]
        p_cur = src[j]
        p_out = out[j]
        for i in range(ncols):
            if i == 0:
                w_index = 1
            else:
                w_index = -1
            cur = p_cur[i]
            w = p_cur[i+w_index]
            n = p_prev[i]
            diff = 0.0
            if (((cur > 0) and (w < 0)) or ((cur < 0) and (w > 0))):
                diff = abs(cur - w)
            if (((cur > 0) and (n < 0)) or ((cur < 0) and (n > 0))):
                vdiff = abs(cur - n)
                if vdiff > diff:
                    diff = vdiff
            if diff > threshold:
                p_out[i] = 255
            else:
                p_out[i] = 0
                
    return out
