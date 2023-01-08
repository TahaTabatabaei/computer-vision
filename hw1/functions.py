import numpy as np

def quantize_simulation(image, n_bits):
    coeff = 2**8 // 2**n_bits
    return (image // coeff) * coeff

def get_bit_planes(image, bit_planes):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = image[i][j] & bit_planes
    image = (((image - image.min()) / (image.max() - image.min())) * 255.0).astype('uint8')
    return image

def calc_hitogram(image):
    length = image.shape[0]
    width = image.shape[1]
    pdf = np.zeros(256)

    for i in range(length):
        for j in range(width):
            k = image[i][j]
            pdf[k] +=1

    return pdf

def normalizeHistogram(pdf,width,length):
    normal_pdf = np.zeros(len(pdf))

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

def mean_square_error(imageSource, imagetarget):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageSource.astype("float") - imagetarget.astype("float")) ** 2)
	err /= float(imageSource.shape[0] * imageSource.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return format(err,'.4f')

def averaging_filter(image,windowSize=3,downsaplingFactor=1):
    #  apply averaging filter with 'windowSize' as filter size and then down-sample the image by a factor of 'downsamplingFactor'
    newWidth = int(image.shape[0]/downsaplingFactor)
    newLength = int(image.shape[1]/downsaplingFactor)
    newImage = np.zeros((newWidth,newLength))

    for i in range(0, image.shape[0], downsaplingFactor):
        for j in range(0, image.shape[1], downsaplingFactor):
            end0 = i+windowSize
            if end0>image.shape[0]:
                end0 = image.shape[0]
            
            end1 = j+windowSize
            if end1>image.shape[1]:
                end1 = image.shape[1]

            buffer = np.mean(image[i:end0, j:end1], axis=(0,1))
            newImage[int(i/downsaplingFactor), int(j/downsaplingFactor)] = buffer

    return newImage

def replication(image,upsamplngFactor=2):
    # upsampling, by copying rows & columns
    newWidth = image.shape[0]*upsamplngFactor
    newLength = image.shape[1]*upsamplngFactor
    newImage = np.zeros((newWidth,image.shape[1]))

    for i in range(image.shape[0]):
        for k in range(upsamplngFactor):
            newImage[(i*upsamplngFactor)+k][:] = image[i][:]

    newImage2 = np.zeros((newWidth,newLength))

    for j in range(newImage.shape[1]):
        for k in range(upsamplngFactor):
            newImage2[: ,(j*upsamplngFactor)+k] = newImage[:,j]
            
    return newImage2
    
def interpolate_bilinear(array_in, array_out):
    width_in = array_in.shape[0]
    height_in = array_in.shape[1]
    width_out = array_out.shape[0]
    height_out = array_out.shape[1]

    for i in range(height_out):
        for j in range(width_out):
            # Relative coordinates of the pixel in output space
            x_out = j / width_out
            y_out = i / height_out

            # Corresponding absolute coordinates of the pixel in input space
            x_in = (x_out * width_in)
            y_in = (y_out * height_in)

            # Nearest neighbours coordinates in input space
            x_prev = int(np.floor(x_in))
            x_next = x_prev + 1
            y_prev = int(np.floor(y_in))
            y_next = y_prev + 1

            # Sanitize bounds - no need to check for < 0
            x_prev = min(x_prev, width_in - 1)
            x_next = min(x_next, width_in - 1)
            y_prev = min(y_prev, height_in - 1)
            y_next = min(y_next, height_in - 1)
            
            # Distances between neighbour nodes in input space
            Dy_next = y_next - y_in
            Dy_prev = 1. - Dy_next; # because next - prev = 1
            Dx_next = x_next - x_in
            Dx_prev = 1. - Dx_next; # because next - prev = 1
            
            # Interpolate over 3 RGB layers
            if (len(array_out.shape) > 2):
                for c in range(3):
                    array_out[i][j][c] = Dy_prev * (array_in[y_next][x_prev][c] * Dx_next + array_in[y_next][x_next][c] * Dx_prev) \
                    + Dy_next * (array_in[y_prev][x_prev][c] * Dx_next + array_in[y_prev][x_next][c] * Dx_prev)
            else: # Interpolate over 1 grayscale layer
                array_out[i,j] = Dy_prev * (array_in[y_next][x_prev] * Dx_next + array_in[y_next][x_next] * Dx_prev) \
            + Dy_next * (array_in[y_prev][x_prev] * Dx_next + array_in[y_prev][x_next] * Dx_prev)
                
    return array_out