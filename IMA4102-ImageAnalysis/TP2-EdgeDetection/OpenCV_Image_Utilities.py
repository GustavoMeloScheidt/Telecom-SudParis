##
## File:	OpenCV_Image_Utilities.py
## Author:	Nicolas ROUGON
## Affiliation:	Institut Polytechnique de Paris | Telecom SudParis | ARTEMIS Department
## Date:	July 31, 2022
##
## Description:	OpenCV sample routines
## > Miscellaneous image utilities
##   - Save image as JPEG or PNG with maximum quality
##   - Read / Write images with file path containing UTF-8 characters
##   - Overlay binary map onto graylevel / color image
##   - Concatenate horizontally two images with arbitrary height
##
## Tutorial
## - Image rasterscan
##   > docs.opencv.org/4.6.0/db/da5/tutorial_how_to_scan_images.html
##

import cv2 as cv
import numpy as np

#
# Save an image as JPEG with maximum quality
#
def save_image_as_JPEG(img, filename):
    flags = [cv.IMWRITE_JPEG_QUALITY, 100]
    cv.imwrite(filename, img, flags)

#
# Save an image as PNG with maximum quality
#
def save_image_as_PNG(img, filename):
    flags = [cv.IMWRITE_PNG_COMPRESSION, 0]
    cv.imwrite(filename, img, flags)

#
# Substitute to OpenCV imread() for images with file path containing UTF-8 characters
# - "flags" must be a valid imread() flag (cv::ImreadModes)
# > docs.opencv.org/4.6.0/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e
# > docs.opencv.org/4.6.0/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80
#
def imread_utf8(filepath, flags):
    img = cv.imdecode(np.fromfile(filepath, dtype="uint8"), flags)
    return img

#
# Substitute to OpenCV imwrite() for images with file path containing UTF-8 characters
# - "extension" must include a leading period "."
#   > docs.opencv.org/4.6.0/d4/da8/group__imgcodecs.html#ga461f9ac09887e47797a54567df3b8b63
# - Compression parameter vector "flags" must be an NumPy array
#   Use np.array([]) if no compression parameters are provided
#

def imwrite_utf8(img, filepath, extension, flags):
    retval, img2 = cv.imencode(extension, img, flags)
    if retval == True:
        img2.tofile(filepath)

#
# Overlay non-zero elements of a scalar image onto a base image
#
def overlay_uchar_image(base, map, color, color3, out):
    # Get input image shape
    if (base.ndim == 2):
        nrows, ncols = np.shape(base)
        nchannels = 1
    else:
        nrows, ncols, nchannels = np.shape(base)
    
    # Get output image ñumber of channels
    if (out.ndim == 2):
        nchannels_out = 1
    else:
        nchannels_out = np.shape(out)[2]

    # Convert map to BGR
    mask = cv.cvtColor(map, cv.COLOR_GRAY2BGR)

    # Change base pixels with non-zero mask values to target color
    if nchannels_out == 1:         # Graylevel overlay
        out = base.copy()      
        out[np.all(mask, axis=-1)] = color
    elif nchannels_out == 3:       # BGR color overlay
        if nchannels == 1:         # Graylevel image
            out = cv.cvtColor(base, cv.COLOR_GRAY2BGR)
        elif nchannels == 3:       # BGR image
            out = base.copy()
        out[np.all(mask, axis=-1)] = color3
    
    return out

