import numpy as np

def dft(input_img):
#     here we get the magnitude od our image 
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    output_img = np.zeros((rows,cols),complex)
    for m in range(0,rows):
        for n in range(0,cols):
            for x in range(0,rows):
                for y in range(0,cols):
#                     this the DFT
                    output_img[m][n] += input_img[x][y] * np.exp(-1j*2*math.pi*(m*x/rows+n*y/cols))
    return output_img

