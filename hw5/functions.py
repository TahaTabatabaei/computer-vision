import numpy as np
import cv2
import math
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from scipy.signal import convolve2d
import pywt

def gaussian_pyramid(image, n_levels=6):
    """
    Compute the Gaussian pyramid

    Inputs:
        - image: Input image of size (N,M)
        - n_levels: Number of stages for the Gaussian pyramid

    Returns:
        Desired gaussian pyramid 
    """

    # approximate length 5 Gaussian filter using binomial filter
    a = 0.4
    b = 1./4
    c = 1./4 - a/2

    filt =  np.array([[c, b, a, b, c]])
    pyr = [image]

    for i in np.arange(n_levels):
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
    """
    Compute the laplacian pyramid

    Inputs:
        - gaussian_pyr: Input gaussian pyramid  

    Returns:
        Desired laplacian pyramid 
    """

    # details pyramid(laplacian)
    # first index should be, the last level of gaussian pyramid
    # alogorithm bulid the pyramid by scaling up the last gaussian pyramid element
    laplacian_top = gaussian_pyr[-1]
    n_levels = len(gaussian_pyr) - 1
    
    laplacian_pyr = [laplacian_top]
    for i in range(n_levels,0,-1):
        # this is the size of gaussian level to be expanded
        # it shuold be equal to index before itself
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])

        # openCV pyrUp make the pyramid transform
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)

        # subtraction operation 
        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr

def pyramid_reconstruct(gaussian_pyr):
    """
    Reconstruct the original image, using provided gaussian (or approximation) pyramid
    NOTE: its really simillar to 'laplacian_pyramid' function, just has one more step;
    that is the summation step, where we add laplacian image and expanded gaussian
    in order to build next level image in the pyramid


    Inputs:
        - gaussian_pyr: Input gaussian pyramid

    Returns:
        Desired laplacian pyramid and reconstructed pyramid
    """


    laplacian_top = gaussian_pyr[-1]
    n_levels = len(gaussian_pyr) - 1
    
    laplacian_pyr = [laplacian_top]

    reconstruct_pyr = [laplacian_top]
    for i in range(n_levels,0,-1):

        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])

        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)

        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)

        # laplacian and expanded gaussian, make next level iamge in reconstruction pyramid
        reconstruct = np.add(laplacian , gaussian_expanded)
        reconstruct_pyr.append(reconstruct)

    return laplacian_pyr , reconstruct_pyr

def box_filter(image,windowSize=3,imagePaddingSize=0):
    """
    Apply averaging filter on Input image. Convolve kernel with size (windoSize,windowSize).

    Inputs:
        - image: Input image of size (N,M)
        - windowSize: Size of averaging kernel
        - imagePaddingSize: Size of image padding. assumed to be eqaul in length and width

    Returns:
        Smoothed image with (windowSize*windowSize) averaging kernel
    """
    # TODO: drop padding

    width = image.shape[0]
    length = image.shape[1]
    size = windowSize-1

    # using 'uint8' to have pixels in range (0,255).
    newImage = np.zeros((width,length),dtype='uint8')

    for i in range(0,length,1):
        for j in range(0,width,1):
            # calculate proper boundaries for window
            # in left and top edges, indexes should be greater than 0
            start = (max(j-size, 0),max(i-size, 0))
            # in right and down edges, indexes should be less than image length and width
            end = (min(j+size, width-1),min(i+size, length-1))

            # crop a part of image which fits to kernel window
            buffer = image.copy()[start[0]:(end[0]+1), start[1]:(end[1]+1)][0]

            # averaging and replacing in the output image
            buffer_mean = np.mean(buffer)
            newImage[j,i] = buffer_mean

    return newImage

def replication(image,zoom_factor=0.5):
    '''
    The nearest neighbor alogorithm, which is the simplest way to interpolate.
    also, shrink image into size, quarter than the input image size.

    Inputs:
        - image: Inpute image of size (N,M)
        - zoom_factor: Shrinking ratio 

    Returns:
        Shirinked image
    '''
    
    # output image size calculation
    newWidth = math.floor(image.shape[1]*zoom_factor) 
    newLength = math.floor(image.shape[0]*zoom_factor) 

    # new image using nearest neighbor method
    return cv2.resize(image ,(newWidth,newLength), interpolation=cv2.INTER_NEAREST)

def approximation_pyramid(image , n_levels):
    """
    Compute the approximation pyramid (gaussian pyramid, but using
    averaging kernel instead of gaussian kernel)

    Inputs:
        - image: Input image of size (N,M)
        - n_levels: Number of stages for the approximation pyramid

    Returns:
        Desired approximation pyramid 
    """

    # first level of pyramid is the original iamge
    level_zero = image.copy()
    approxi_pyr = [level_zero]


    for _ in range(n_levels):

        # applying averaging filter
        level_zero = box_filter(level_zero,windowSize=2)

        # applying replication for interpolation
        level_zero = replication(level_zero)

        approxi_pyr.append(np.asarray(level_zero))
    
    return approxi_pyr

def mean_square_error(imageSource, imagetarget):
    """
	The "Mean Squared Error" between the two images is the
	sum of the squared difference between the two images.
    the lower the error, the more "similar" the two images are.
    
	NOTE: the two images must have the same dimension

    Inputs:
        - imageSource: The source image, we want to calculate the target image difference of 
        - imageTarget: The target image, we calculate how far it is from the source
	
	Returns:
        The MSE
    """

    # cumulative difference 
    err = np.sum((imageSource.astype("float") - imagetarget.astype("float")) ** 2)

    # divide by length*width
    err /= float(imageSource.shape[0] * imageSource.shape[1])
	
    return format(err,'.6f')

def coefficientQuantizer(coeff,step=2):
    """
    Quantize the wavelet coeddicients, using the formula in the exercise description.
    The formula, simply divide coefficients by given 'step', and round them, and again
    multiply by 'step'. in this way, we have quantized coefficients.

    Inputs:
        - coeff: Given coefficients to be quantized
        - step: Scale of quantization
    
    Returns:
        New coefficients
    """
    coeff_new = step * np.sign(coeff) * np.floor(np.abs(coeff)/step)
    return coeff_new

def PSNR(srcImage,testImage):
    """
    Implementation of 'Peak Signal to Noise Ratio' method
    using, sci-kit image library.
    The greater the result, the more "similar" the two images are.

    Inputs:
        - srcImage: The source image, we want to calculate the target image difference of 
        - testImage: The target image, we calculate how similar it is with the source    

    Returns:
        The PSNR
    """
    return peak_signal_noise_ratio(srcImage,testImage)

def wavelet_payramid(image,n_levels,normalization=False):
    """
    Computing wavelet pyramid, using haar method.

    NOTE: if you normalize the coefficients (by setting 'normalization=False') you
    can not rebnild the original image properly. In order to rebuild the original image you
    need to transform the coefficients back to the original domain.

    Inputs:
        - image: Provided image we want to decomposite.
        - n_levels: Number of stages for wavelet pyramid
        - normalization: A flag, which consider normalization over coefficients

    Returns:
        Desired wavelet pyramid in form of array, Desired wavelet pyramid in form of list
    
    """

    coeffs = pywt.wavedec2(image, 'haar', mode='periodization', level=n_levels,)
    if normalization:
        coeffs = normalize(coeffs,1,0)
    c_matrix, c_slices = pywt.coeffs_to_array(coeffs)

    return c_matrix , coeffs

def normalize(array,newMax,newMin):
    """
    A simple normalization function.

    Inputs:
        - array: Array to be normalized
        - newMax: Max of new range.
        - newMin: Min of new range.

    Returns:
        Normalized array.
    
    """
    if isinstance(array, list):
        return list(map(normalize, array,newMax,newMin))
    if isinstance(array, tuple):
        return tuple(normalize(list(array),newMax,newMin))
    normalizedData = (array-np.min(array))/(np.max(array)-np.min(array))*(newMax-newMin) + newMin
    return normalizedData
