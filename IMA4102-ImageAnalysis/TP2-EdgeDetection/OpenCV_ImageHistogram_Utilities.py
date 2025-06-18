##
## File:	OpenCV_ImageHistogram_Utilities.py
## Author:	Nicolas ROUGON
## Affiliation:	Institut Polytechnique de Paris | Telecom SudParis | ARTEMIS Department
## Date:	July 31, 2022
##
## Description:	OpenCV sample routines
## > Histogram display utilities
##
## Documentation
## - Array normalization
##   > docs.opencv.org/4.6.0/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd
## - Graphics | Draw a circle
##   > docs.opencv.org/4.6.0/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
##
## Tutorial
## - Histogram calculation
##   > docs.opencv.org/4.6.0/d8/dbc/tutorial_histogram_calculation.html
##

import cv2 as cv
import numpy as np

#
# Compute image histogram and render it as an image with predefined size
#

def computeRenderHistogram(image, nb_bins, histRange, bin_ranges, bin_colors,
                            nb_ranges, draw_mode, image_hist):
    # Compute image histogram
    hist = cv.calcHist([image], [0], None, [nb_bins], histRange, False)

    # Draw histogram image
    image_hist = np.zeros(np.shape(image_hist), dtype="uint8")
    image_hist = renderHistogram(hist, nb_bins, bin_ranges, bin_colors, nb_ranges,
                                 draw_mode, image_hist)

    return image_hist

#
# Render histogram as an image with predefined size
#
def renderHistogram(hist, nb_bins, bin_ranges, bin_colors, nb_ranges, draw_mode, image_hist):    
    # Rescale histogram into [0, image_hist.rows] for display
    rows = np.shape(image_hist)[0]
    cv.normalize(hist, hist, 0, rows, cv.NORM_MINMAX, -1, None)
    
    # Draw histogram
    drawHistogram(hist, nb_bins, bin_ranges, bin_colors, nb_ranges, draw_mode, image_hist)

    return image_hist

#
# Render histogram as an image 
#
def drawHistogram(hist, nb_bins, bin_ranges, bin_colors, nb_ranges, draw_mode, image_hist):
    rows, cols, channels = np.shape(image_hist)
    bin_width = round((float)(cols)/nb_bins)
    
    if draw_mode == 1:       # Render as bar chart
        for j in range(nb_ranges):
            for i in bin_ranges[j]:
                cv.rectangle(image_hist, 
                             (bin_width*(i), rows - round(hist[i][0])),
                             (bin_width*(i+1), rows), 
                             bin_colors[j], cv.FILLED, 8, 0)
    elif draw_mode == 2:     # Render as curve
        for j in range(nb_ranges):
            for i in bin_ranges[j]:
                cv.line(image_hist, 
                        (bin_width*(i), rows - round(hist[i][0])),
                        (bin_width*(i+1), rows - round(hist[i+1][0])),
                        bin_colors[j], 2, 8, 0);
