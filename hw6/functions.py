import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio


def RGB2HSI(image):
    with np.errstate(divide='warn', invalid='warn'):

        # normalize pixels in range [0,1]
        image_normal = np.float32(image)/255.0

        #Separate color channels
        # typr:ignore
        red = image_normal[:,:,0]
        green = image_normal[:,:,1]
        blue = image_normal[:,:,2]



        

        h = hue(red,green,blue)
        # h = calc_hue(red,green,blue)
        s = saturation(red,green,blue)
        i = intensity(red,green,blue)


        return h,s,i


def hue(red, green, blue):
    """
    Calculate the hue(𝐻) in the HSI color space from the RGB channels. Hue is calculated
    according to the following formula.
    
    𝐻 = 
    𝜃 
        if (B ≤ G)    
        ,else
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

    numerator = np.multiply(0.5,(np.add(np.subtract(red,green),np.subtract(red,blue))))


    z1 = np.add(np.power(np.subtract(red,green),2), np.multiply(np.subtract(red,green),np.subtract(green,blue)))
    z1 = np.where(((z1-0) <= 0.00001) , 0.00001 , z1)
    denominator = np.sqrt(z1)

    x = np.divide(numerator,denominator)
    # print(f'x = {x}')

    tetha = np.arccos(np.divide(np.multiply(x,(math.pi*2)), 360))
    # tetha = np.arccos(x)
    # print(f'tetha2 = {tetha}')

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
    # saturation = 1 - ((3 / (red + green + blue + 0.0001)) * minimum)
    saturation = np.subtract(1,np.multiply(np.divide(3,np.add(np.add(red,blue,green),0.0001)),minimum))

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

#Calculate Hue
def calc_hue(red,green,blue):
    hue = np.copy(red)

    for i in range(0, blue.shape[0]):
        for j in range(0, blue.shape[1]):
            hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                        math.sqrt((red[i][j] - green[i][j])**2 +
                                ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
            hue[i][j] = math.acos(hue[i][j])

            # print(f'tetha = {hue[i][j]}')

            if blue[i][j] <= green[i][j]:
                hue[i][j] = hue[i][j]
            else:
                hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]
            

    return hue


def normalize(array,newMax,newMin):
    if isinstance(array, list):
        return list(map(normalize, array,newMax,newMin))
    if isinstance(array, tuple):
        return tuple(normalize(list(array),newMax,newMin))
    normalizedData = (array-np.min(array))/(np.max(array)-np.min(array))*(newMax-newMin) + newMin
    return normalizedData


def quantize(array, n_bits):
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