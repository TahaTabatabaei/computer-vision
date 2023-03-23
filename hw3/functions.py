import numpy as np
import random as rand
import cv2


def clip_filter(image,paddingSize):
    """
    Apply the "clip filter" method of padding an image, which adds zero padding around the 
    input image.

    Inputs:
        - image: Input image of size (N,M)
        - paddingSize: Desired size of image padding. Assumed to be eqaul in length and width.

    Returns:
        Image with padding.
    """

    # the new image: image + padding
    frame = np.zeros((image.shape[0]+int(2*paddingSize) ,image.shape[1]+int(2*paddingSize) ,3),dtype='uint8')
    
    width = frame.shape[0]
    length = frame.shape[1]

    # copy the source image to the output frame.
    for i in range(paddingSize,length-paddingSize):
        for j in range(paddingSize,width-paddingSize):
            x= i-paddingSize 
            y= j-paddingSize
            frame[j][i] = image.copy()[y][x]
    
    return frame


def box_filter(image,windowSize=3,imagePaddingSize=0):
    """
    Apply averaging filter on Input image. Convolve kernel with size (windoSize,windowSize).

    Inputs:
        - image: Input image of size (width,length)
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

    for i in range(0,width,1):
        for j in range(0,length,1):

            # calculate proper boundaries for window
            # in left and top edges, indexes should be greater than 0
            start = (max(i-size, 0),max(j-size, 0))
            # in right and down edges, indexes should be less than image length and width
            end = (min(start[0]+size, width-1),min(start[1]+size, length-1))

            # crop a part of image which fits to kernel window
            buffer = image.copy()[start[0]:(end[0]+1), start[1]:(end[1]+1)]
                      
            # averaging and replace in the output image
            buffer_mean = np.mean(buffer)
            newImage[i,j] = buffer_mean

    if imagePaddingSize>0:
        return newImage.copy()[imagePaddingSize:width-imagePaddingSize, imagePaddingSize:length-imagePaddingSize]
    return newImage
            

def laplacian_filter(image,mask):
    """
    An special type of "weighted filter" is used. The input mask should satisfy the Laplacian filter
    constraints, so this function is simply calling the default "weighted filter".

    Inputs:
        - image: Input image of size (width,length)
        - mask: Mask to be convolved with the image.

    Returns:
        The convolution result.
    
    """
    return weighted_filter(image,mask)


def salt_pepper(image, noiseDensity):
    """
    The salt & papper noise generation mehtod. In this method, we add random salt
    (pixels with 255 intensity) and papper(pixels with 0 intensity). The greater the 
    "noiseDensity", the more salt & pepper generation results.

    Inputs:
        - image: Input image of size (N,M).
        - noiseDensity: What percentage of the image should be turned to noise.

    Returns:
        The noisy image.
    """

    # calculate number of noisy pixels we need.
    n_noise = image.shape[0]*image.shape[1]*noiseDensity

    for i in range(int(n_noise)):
        # salt or papper
        k = rand.randrange(2)
        if k == 0 :
            # papper
            gray = 0
        else:
            # salt
            gray = 255

        # generate a random x,y. if the pixel in these coordinates is already 0 or 255, we
        # regenerate another x,y . but if it is not 0 or 255, we break the while loop so it
        # can be turned to salt or pepper.
        while(True):
            x = rand.randrange(image.shape[0])
            y = rand.randrange(image.shape[1])
            if len(image.shape) == 3:
                # for images in RGB
                if (gray != image[x][y][0]) or (gray != image[x][y][1]) or (gray != image[x][y][2]):
                    break
            else:
                # for gray-scale
                if gray != image[x][y]:
                    break

        image[x][y] = gray
        
    return image

def median_filter(image, windowSize=3,imagePaddingSize=0):
    """
    Apply the median to each window while iterating over the input image. Convolve the kernel
    with a size of (windowSize, windowSize).

    Inputs:
        - image: Input image of size (width,length)
        - windowSize: Size of median kernel
        - imagePaddingSize: Size of image padding. assumed to be eqaul in length and width

    Returns:
        Smoothed image with (windowSize*windowSize) median kernel.
    """

    width = image.shape[0]
    length = image.shape[1]
    size = windowSize-1

    # using 'uint8' to have pixels in range (0,255).
    newImage = np.zeros((width,length),dtype='uint8')

    for i in range(0,width,1):
        for j in range(0,length,1):

            # calculate proper boundaries for window
            # in left and top edges, indexes should be greater than 0
            start = (max(i-size, 0),max(j-size, 0))
            # in right and down edges, indexes should be less than image length and width
            end = (min(start[0]+size, width-1),min(start[1]+size, length-1))

            # crop a part of image which fits to kernel window
            buffer = image.copy()[start[0]:(end[0]+1), start[1]:(end[1]+1)]

            # calculate median and replace in output image
            buffer_median = np.median(buffer)
            newImage[i][j] = buffer_median
    
    return newImage

def weighted_filter(image,mask,weight=1):
    """
    A generalization for a appling mask with weights in the form of:
        weight = 1 / (absolut sum of mask elements)
    The weight is actually for normalization, so you can ignore it, if you are
    choosing mask elements consciously. 
    The function supports mask with different widths.

    Inputs:
        - image: Input image of size (width,length)
        - mask: Mask to be convolved with the image.
        - weight: The formoula = 1 / (absolut sum of mask elements) . The default value
        is 1, so it does not effect the result.

    Returns:
        The convolution result.



    """
    width = image.shape[0]
    length = image.shape[1]
    wsize = mask.shape[0]-1
    lsize = mask.shape[1]-1


    # using 'uint8' to have pixels in range (0,255).
    newImage = np.zeros((width,length),dtype='uint8')

    for i in range(0,width,1):
        for j in range(0,length,1):

            # calculate proper boundaries for window
            # in left and top edges, indexes should be greater than 0
            start = (max(i-wsize, 0),max(j-lsize, 0))
            # in right and down edges, indexes should be less than image length and width
            end = (min(start[0]+wsize, width-1),min(start[1]+lsize, length-1))

            # crop a part of image which fits to kernel window
            buffer = image.copy()[start[0]:(end[0]+1), start[1]:(end[1]+1)]
                      
            # image and mask convolution(element-wise multiply)
            result_matrix = np.multiply(buffer,mask)
            
            result = 0
            for m in range(result_matrix.shape[0]):
                for n in range(result_matrix.shape[1]):
                    result += result_matrix[m][n]
            
            if result>255:
                result = 255
            elif result<0:
                result = 0

            x = int((end[0]-start[0])/2  + start[0]) 
            y = int((end[1]-start[1])/2 + start[1])
            newImage[x][y] = result*weight

    return newImage

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



# TODO: to vaildate inputs