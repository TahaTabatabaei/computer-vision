import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def calc_histogram(image):
    """
    Calculate image histogram. Iterates over an image and counts how meny times
    it sees a pixel with an intensity of "k". "k" is in range [0,255]. "pdf" stands
    for Probability Density Function.

    Inputs:
        - image: Input image of size (width,length)

    Returns:
        The image histogram.
    """
    width = image.shape[0]
    length = image.shape[1]

    pdf = np.zeros(256)

    # counts the intensity frequency
    for i in range(width):
        for j in range(length):
            k = image[i][j]
            pdf[k] +=1


    return pdf

def normalizeHistogram(pdf,width,length):
    """
    A simple normalize on the histogram(pdf).

    Inputs:
        - pdf: A histogram.
        - width: Width of image we wnat to normalize its histogram.
        - length: length of image we wnat to normalize its histogram.

    Returns:
        Normalized histogram. 
    """
    normal_pdf = np.zeros(len(pdf))

    # normalize step
    for i in range(len(normal_pdf)):
        normal_pdf[i] = pdf[i]/(width*length)

    return normal_pdf

def calc_cdf(normal_pdf):
    """
    Calculates the CDF(Cumulative distribution function) over the input histogram. 

    Inputs:
        - normal_pdf: Normalized pdf.

    Returns:
        The CDF of input histogram.
    """
    cdf = np.zeros(len(normal_pdf))

    # CDF calculation.
    for i in range(len(normal_pdf)):
        for j in range(i):
            cdf[i] += normal_pdf[j]
    
        # revert to range [0,255]
        cdf[i] *=255
        cdf[i] = round(cdf[i])
    
    return cdf

def reMap(image,target_cdf):
    """
    Update the image pixels' intensity based on the target CDF. This function
    implements a look-up table.

    Inputs:
        - image: Input image of size (N,M)
        - target_cdf: A CDF table we want to use for equalization.

    Returns;
        An image whose pixels have been changed in order to achieve greater diversity.
    """
    newImage = image.copy()
    for i in range(newImage.shape[0]):
        for j in range(newImage.shape[1]):
            newImage[i][j] = target_cdf[newImage[i][j]]

    return newImage

def log_transform(image, c):
    """
    Logarithmic transform. This transformation expands the dark pixels in the image
    while compressing the brighter pixels.

    Inputs: 
        - image: Input image of size (width,length)
        - c: Arbitrary constant

    Returns:
        The changed image.
    """

    # the logarithmic transform formula 
    s = c*np.log(1+image)
    return s

def inverse_log_transform(image , c, base):
    """
    Inverse logarithmic transform. This transformation expands the bright pixels in the image
    while compressing the darker pixels.

    Inputs: 
        - image: Input image of size (width,length)
        - c: Arbitrary constant

    Returns:
        The changed image.
    """
    # the inverse logaritmic transform formula 
    s = c*(np.power(base,image)-1)
    return s

def power_law_transform(image, c, gamma):
    """
    Power Law (Gamma) transform. Gamma correction is important for displaying images on
    a screen correctly, to prevent bleaching or darkening of images. For gamma in range (0,1) , you can consider
    this function as « n-th root transform ».

    Inputs: 
        - image: Input image of size (width,length)
        - c: Arbitrary constant

    Returns:
        The changed image.
    """

    # the power law transform formula 
    s = c*np.power(image,gamma)
    return s


def local_histo_equalization(image,windowSize):
    """
    Applying a local histogram equalization routine, this function calls methods previously
    implemented in order to achieve a valid set of actions.

    Inputs:
        - image: Input image of size (width,length)
        - windowSize: Size of kernel

    Returns:
        Locally equalized image.
    """

    width , length , band =image.shape
    size = windowSize-1

    newImage = np.zeros_like(image)

    for i in range(0,width+size,windowSize):
        for j in range(0,length+size,windowSize):
            # calculate proper boundaries for window
            # in left and top edges, indexes should be greater than 0
            start = (max(i-size, 0),max(j-size, 0))
            # in right and down edges, indexes should be less than image length and width
            end = (min(start[0]+size, width-1),min(start[1]+size, length-1))

            # crop a part of image which fits to kernel window
            buffer = image.copy()[start[0]:(end[0]+1), start[1]:(end[1]+1)]

            # equalization routine
            buffer_histo = calc_histogram(buffer)
            normal_buffer = normalizeHistogram(buffer_histo,((end[0]+1)-start[0]),((end[1]+1)-start[1]))
            buffer_cdf = calc_cdf(normal_buffer)
            remaped_buffer = reMap(buffer,buffer_cdf)

            # replace in the output image
            newImage[start[0]:(end[0]+1), start[1]:(end[1]+1)] = remaped_buffer

    return newImage

def histeq(image):
    """
    Histogram equalization routine.

    Inputs:
        - image: Input image of size (width,length)

    Returns:
        Equalized image.
    """
    width ,length ,band = image.shape
    newImage = np.zeros_like(image)

    # equalization routine
    buffer_histo = calc_histogram(image)
    normal_buffer = normalizeHistogram(buffer_histo,width,length)
    buffer_cdf = calc_cdf(normal_buffer)
    remaped_buffer = reMap(image,buffer_cdf)

    # replace in the output image
    newImage = remaped_buffer
    return newImage
    
