import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio


def RGB2HSI(image):
    """
    Transform image from RGB space to HSI. Calculates Hue, Saturation & Intensity.

    Inputs:
        - image: The source image.

    Returns:
        Image in HSI space.
    """

    # suspend errors: 1) divide by zero 2) inavalid data-type
    with np.errstate(divide='warn', invalid='warn'):

        # normalize pixels in range [0,1]
        image_normal = np.float32(image)/255.0

        # Separate color channels
        # typr:ignore
        red = image_normal[:,:,0]
        # type: ignore      
        green = image_normal[:,:,1]
        # type: ignore 
        blue = image_normal[:,:,2]

        h = hue(red.copy(),green.copy(),blue.copy())

        s = saturation(red.copy(),green.copy(),blue.copy())

        i = intensity(red.copy(),green.copy(),blue.copy())

        return h,s,i


def hue(red, green, blue):
    """
    Calculate the hue(𝐻) in the HSI color space from the RGB channels. Hue is calculated
    according to the following formula.
    
    𝐻 = 
    𝜃 
        if (B ≤ G)    
        ,else:
            (360 − 𝜃) 


    where:    

    𝜃 = arccosine
        {
        1/2 * [(𝑅 − 𝐺) + (𝑅 − 𝐵)]
        /
        sqrt( power((𝑅 − 𝐺),2) + (𝑅 − 𝐵)*(𝐺 − 𝐵) )
        }

    Inputs:
        - red: Red channel in RGB space.
        - green: Green channel in RGB space.
        - blue: Blue channel in RGB space.

    Returns:
        The hue.

    """

    # numerator of equation
    numerator = np.multiply(0.5,(np.add(np.subtract(red,green),np.subtract(red,blue))))

    # denominator of equation
    z1 = np.add(np.power(np.subtract(red,green),2), np.multiply(np.subtract(red,green),np.subtract(green,blue)))
    z1 = np.where(((z1-0) <= 0.00001) , 0.00001 , z1)
    denominator = np.sqrt(z1)

    # final division
    x = np.divide(numerator,denominator)

    # tetha in range [0,2*pi]
    tetha = np.arccos(np.divide(np.multiply(x,(math.pi*2)), 360))

    # if B <= G  h = tetha , else h = 2*pi - tetha
    hue = np.where(blue<=green , tetha , (math.pi*2.0 - tetha))

    return hue


def saturation(red, green, blue):
    """
    Calculate the saturation in the HSI color space from the RGB channels. Saturation is
    calculated according to the formula below.
    
    𝑆 = 1 − ( 3 * [min( 𝑅, 𝐺, 𝐵)] / (𝑅 + 𝐺 + 𝐵) )

    Inputs:
        - red: Red channel in RGB space.
        - green: Green channel in RGB space.
        - blue: Blue channel in RGB space.

    Returns:
        The saturation.

    """
    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = np.subtract(1,np.multiply(np.divide(3,np.add(np.add(red,blue,green),0.0001)),minimum))
    # saturation = 1 - np.multiply(np.divide(3,np.add(red,blue,green)),minimum)

    return saturation


def intensity(red, green, blue):
    """
    Calculate the intensity in the HSI color space from the RGB channels. Intensity is
    the average value of the three channels.
    
    Inputs:
        - red: Red channel in RGB space.
        - green: Green channel in RGB space.
        - blue: Blue channel in RGB space.

    Returns:
        The Intensity.
    """

    return np.divide((blue + green + red), 3)

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


def quantize(array, n_bits):
    """
    Image array(range of 0 to 255) is quantized to the desired number of bits. Its actually generate 
    a new array, assuming that the representation uses 'n_bits' instead of the default 8 bits.

    Inputs:
        - array: The image matrix
        - n_bits: Given number of bits.

    Returns:
        The quantized array.
    """
    coeff = 2**8 // 2**n_bits
    return (array // coeff) * coeff


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