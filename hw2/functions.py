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
        - Input image of size (width,length)

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

    cdf = np.zeros(len(normal_pdf))

    for i in range(len(normal_pdf)):
        for j in range(i):
            cdf[i] += normal_pdf[j]
    
        cdf[i] *=255
        cdf[i] = round(cdf[i])
    
    return cdf

def reMap(image,target_cdf):
    newImage = image.copy()
    for i in range(newImage.shape[0]):
        for j in range(newImage.shape[1]):
            newImage[i][j] = target_cdf[newImage[i][j]]

    return newImage

def log_transform(image, c):
    s = c*np.log(1+image)
    return s

def inverse_log_transform(image , c, base):
    s = c*(np.power(base,image)-1)
    return s

def power_law_transform(image, c, gamma):
    s = c*np.power(image,gamma)
    return s


def local_histo_equalization(image,windowSize):
    width , length , band =image.shape

    newImage = np.zeros_like(image)
    n = 0
    for i in range(0,length,windowSize):
        for j in range(0,width,windowSize):
            start = (j,i)
            end0 = j+windowSize
            end1 = i+windowSize
            
            if end0>width:
                end0 = width
                
            if end1>length:
                end1 = length
            
            end = (end0,end1)
            
            buffer = image[start[0]:end[0], start[1]:end[1]]

            buffer_histo = calc_histogram(buffer)
            normal_buffer = normalizeHistogram(buffer_histo,windowSize,windowSize)
            buffer_cdf = calc_cdf(normal_buffer)
            remaped_buffer = reMap(buffer,buffer_cdf)

            newImage[start[0]:end[0], start[1]:end[1]] = remaped_buffer

            n +=1
            
    eqaulizedImage = newImage
    return eqaulizedImage