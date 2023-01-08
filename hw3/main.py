import numpy as np
import random
from PIL import Image

'''
filter ha 3*3 hastand baraye emale harkodum, zarayeb
an dar pixel ha zarb shode va sepas natije dar pixel
markazi rikhte mishavad.

filter gaussian va miangin yeknavakht tasvir ra kami tar mikonand
vali az shar noiz khalas nemishavand, filter miane be behtarin shekl
noise ra pak mikonad. filter maximum noise haye sefid va filter minimum 
noise haye siah ro bozorg tar karde va noise digar ra pak mikonand.
'''

numpydata = np.array(Image.open('histogram.jpeg'))

#besurat random noghati az tasvir entekhab shode va
#sefid yasiah mishavand
noiseN = 10000
for i in range(noiseN):
    x = random.randint(0, numpydata.shape[1]-1)
    y = random.randint(0, numpydata.shape[0]-1)
    numpydata[y][x] = 0
    x = random.randint(0, numpydata.shape[1]-1)
    y = random.randint(0, numpydata.shape[0]-1)
    numpydata[y][x] = 255
im = Image.fromarray(numpydata)
Image._show(im)

#filter miangin yeknavakht
blurImg = numpydata.copy()
for i in range(1, numpydata.shape[0]-1):
    for j in range(1,numpydata.shape[1]-1):
        sum = int(numpydata[i-1][j-1][0]) + int(numpydata[i-1][j][0]) + int(numpydata[i-1][j+1][0]) + \
            int(numpydata[i][j-1][0]) + int(numpydata[i][j][0]) + int(numpydata[i][j+1][0]) + \
            int(numpydata[i+1][j-1][0]) + int(numpydata[i+1][j][0]) + int(numpydata[i+1][j+1][0])
        blurImg[i][j] = int(sum / 9)

im = Image.fromarray(blurImg)
Image._show(im)

#filter Gaussian sigma=1
blurImg = numpydata.copy()
for i in range(1, numpydata.shape[0]-1):
    for j in range(1,numpydata.shape[1]-1):
        sum = int(numpydata[i-1][j-1][0]) + 2*int(numpydata[i-1][j][0]) + int(numpydata[i-1][j+1][0]) + \
            2*int(numpydata[i][j-1][0]) + 4*int(numpydata[i][j][0]) + 2*int(numpydata[i][j+1][0]) + \
            int(numpydata[i+1][j-1][0]) + 2*int(numpydata[i+1][j][0]) + int(numpydata[i+1][j+1][0])
        blurImg[i][j] = int(sum / 16)

im = Image.fromarray(blurImg)
Image._show(im)

#filter miane
blurImg = numpydata.copy()
sortedList = []
for i in range(1, numpydata.shape[0]-1):
    for j in range(1,numpydata.shape[1]-1):
        sortedList.extend([int(numpydata[i-1][j-1][0]), int(numpydata[i-1][j][0]), int(numpydata[i-1][j+1][0]),
                           int(numpydata[i][j-1][0]), int(numpydata[i][j][0]), int(numpydata[i][j+1][0]),
                           int(numpydata[i+1][j-1][0]), int(numpydata[i+1][j][0]), int(numpydata[i+1][j+1][0])] )
        sortedList.sort()
        blurImg[i][j] = sortedList[4]
        sortedList.clear()
im = Image.fromarray(blurImg)
Image._show(im)

#filter maximum
blurImg = numpydata.copy()
sortedList = []
for i in range(1, numpydata.shape[0]-1):
    for j in range(1,numpydata.shape[1]-1):
        sortedList.extend([int(numpydata[i-1][j-1][0]), int(numpydata[i-1][j][0]), int(numpydata[i-1][j+1][0]),
                           int(numpydata[i][j-1][0]), int(numpydata[i][j][0]), int(numpydata[i][j+1][0]),
                           int(numpydata[i+1][j-1][0]), int(numpydata[i+1][j][0]), int(numpydata[i+1][j+1][0])] )
        sortedList.sort()
        blurImg[i][j] = sortedList[8]
        sortedList.clear()
im = Image.fromarray(blurImg)
Image._show(im)

#filter minimum
blurImg = numpydata.copy()
sortedList = []
for i in range(1, numpydata.shape[0]-1):
    for j in range(1,numpydata.shape[1]-1):
        sortedList.extend([int(numpydata[i-1][j-1][0]), int(numpydata[i-1][j][0]), int(numpydata[i-1][j+1][0]),
                           int(numpydata[i][j-1][0]), int(numpydata[i][j][0]), int(numpydata[i][j+1][0]),
                           int(numpydata[i+1][j-1][0]), int(numpydata[i+1][j][0]), int(numpydata[i+1][j+1][0])] )
        sortedList.sort()
        blurImg[i][j] = sortedList[0]
        sortedList.clear()
im = Image.fromarray(blurImg)
Image._show(im)
