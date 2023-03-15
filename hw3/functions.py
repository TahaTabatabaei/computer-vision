import numpy as np
import random as rand
import cv2

# add padding to image
def clip_filter(image,paddingSize):
    frame = np.zeros((image.shape[0]+int(2*paddingSize) ,image.shape[1]+int(2*paddingSize) ,3),dtype='uint8')
    width = frame.shape[0]
    length = frame.shape[1]

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

    print(size)

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

    return newImage
            

def laplacian_filter(image,mask):
    return weighted_filter(image,mask,weight=1)


def salt_pepper(image, noiseDensity):
    n_noise = image.shape[0]*image.shape[1]*noiseDensity
    for i in range(int(n_noise)):
        k = rand.randrange(2)
        if k == 0 :
            gray = 0
        else:
            gray = 255

        while(True):
            x = rand.randrange(image.shape[0])
            y = rand.randrange(image.shape[1])
            if len(image.shape) == 3:
                if gray != image[x][y][0]:
                    break
            else:
                if gray != image[x][y]:
                    break

        image[x][y] = gray
        
    return image

def median_filter(image, windowSize=3,imagePaddingSize=0):
    # TODO: make it in the form of box filter inside the loops
    width = image.shape[0]
    length = image.shape[1]
    size = windowSize-1

    print(size)

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
    # TODO: its faulty, resolve issue
    width = image.shape[0]
    length = image.shape[1]
    size = mask.shape[0]-1

    print(size)

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
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageSource.astype("float") - imagetarget.astype("float")) ** 2)
	err /= float(imageSource.shape[0] * imageSource.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return format(err,'.4f')