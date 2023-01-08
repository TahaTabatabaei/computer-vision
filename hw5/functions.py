import numpy as np
import cv2
import math
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from scipy.signal import convolve2d

# def gaussian_pyramid(image, n_levels):
#     level_zero = image.copy()
#     gaussian_pyr = [level_zero]
#     for _ in range(n_levels):
#         level_zero = cv2.pyrDown(level_zero)
#         gaussian_pyr.append(np.float32(level_zero))
#     return gaussian_pyr

def gaussian_pyramid(img, levels=6):
    """
    Compute the Gaussian pyramid

    Inputs:
    - img: Input image of size (N,M)
    - levels: Number of stages for the Gaussian pyramid

    Returns:
    A tuple of levels images 
    """

    # approximate length 5 Gaussian filter using binomial filter

    a = 0.4
    b = 1./4
    c = 1./4 - a/2

    filt =  np.array([[c, b, a, b, c]])

    # approximate 2D Gaussian
    # filt = convolve2d(filt, filt.T)

    pyr = [img]

    for i in np.arange(levels):
        # zero pad the previous image for convolution
        # boarder of 2 since filter is of length 5
        p_0 = np.pad( pyr[-1], (2,), mode='constant' )

        # convolve in the x and y directions to construct p_1
        p_1 = convolve2d( p_0, filt, 'valid' )
        p_1 = convolve2d( p_1, filt.T, 'valid' )

        # DoG approximation of LoG
        pyr.append( p_1[::2,::2] )

    return pyr

def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    n_levels = len(gaussian_pyr) - 1
    
    laplacian_pyr = [laplacian_top]
    for i in range(n_levels,0,-1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        # print(size)
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr

def pyramid_reconstruct(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    n_levels = len(gaussian_pyr) - 1
    
    laplacian_pyr = [laplacian_top]

    reconstruct_pyr = [laplacian_top]
    for i in range(n_levels,0,-1):
        # size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        # print(size)

        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)

        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)

        reconstruct = np.add(laplacian , gaussian_expanded)
        reconstruct_pyr.append(reconstruct)

    return laplacian_pyr , reconstruct_pyr

def box_filter(image,windowSize=3,imagePaddingSize=0):
    width = image.shape[0]
    length = image.shape[1]
    size = windowSize-1
    newImage = np.zeros((width-(2*imagePaddingSize),length-(2*imagePaddingSize)),dtype='uint8')

    for i in range(0,length,1):
        for j in range(0,width,1):
            start = (max(j-size, 0),max(i-size, 0))
            # end0 = j+windowSize
            # end1 = i+windowSize
            
            # if end0>width:
            #     end0 = width
                
            # if end1>length:
            #     end1 = length
            
            end = (min(j+size, width-1),min(i+size, length-1))
            # x_start = max(i-size, 0)
            # x_end = min(i+size, img.shape[0]-1)
            # y_start = max(j-size, 0)
            # y_end = min(j+size, img.shape[1]-1)
    
            # value = img[x_start:x_end+1, y_start:y_end+1].mean()
            buffer = image.copy()[start[0]:(end[0]+1), start[1]:(end[1]+1)][0]
            buffer_mean = np.mean(buffer)
            newImage[j,i] = buffer_mean

    return newImage

# TO DO: diffrent factors of zoom or shrink
def replication(image,zoom_factor=0.5):
    newWidth = math.floor(image.shape[1]*zoom_factor) 
    newLength = math.floor(image.shape[0]*zoom_factor) 
    return cv2.resize(image ,(newWidth,newLength), interpolation=cv2.INTER_NEAREST)

def approximation_pyramid(image , n_levels):
    level_zero = image.copy()
    approxi_pyr = [level_zero]
    for _ in range(n_levels):
        level_zero = box_filter(level_zero,windowSize=2)
        level_zero = replication(level_zero)
        # print(level_zero.shape)
        approxi_pyr.append(np.asarray(level_zero))
    
    return approxi_pyr

    #     level_zero = image.copy()
    # gaussian_pyr = [level_zero]
    # for i in range(n_levels):
    #     level_zero = cv2.pyrDown(level_zero)
    #     gaussian_pyr.append(np.float32(level_zero))


def mean_square_error(imageSource, imagetarget):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageSource.astype("float") - imagetarget.astype("float")) ** 2)
	err /= float(imageSource.shape[0] * imageSource.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return format(err,'.6f')

def coefficientQuantizer(coeff,step=2):
    coeff_new = step * np.sign(coeff) * np.floor(np.abs(coeff)/step)
    return coeff_new

def PSNR(srcImage,testImage):
    return peak_signal_noise_ratio(srcImage,testImage)
