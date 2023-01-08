#  making array
windowSize = 16
n_parts = 1024
images = np.zeros((n_parts, windowSize, windowSize, 3), dtype=int)



# segment image into 16*16 parts
print(type(images))
n = 0
for i in range(0,width,windowSize):
    for j in range(0,length,windowSize):
        start = (j,i)
        end = (j+windowSize,i+windowSize)
        # print(n)
        images[n] = image.copy()[start[0]:end[0], start[1]:end[1]]
        # if n%35 == 0:
        #     plt.imshow(images[n])
        n +=1
# calculate histogram for each part
local_pdfs = [np.zeros(256)]*n_parts
# print(local_pdfs[0])
for i in range(len(images)):
    local_pdfs[i] = calc_hitogram(images[i])


# print(local_pdfs[0])
# normalize each part
normal_local_pdfs = []

for local_pdf in local_pdfs:
    normal_local_pdfs.append(normalizeHistogram(local_pdf,windowSize,windowSize))

# normal_local_pdfs[0]
# calculate cdf for each part
local_cdfs = []

for nLocal_pdf in normal_local_pdfs:
    # print(nLocal_pdf)
    local_cdfs.append(calc_cdf(nLocal_pdf))

# local_cdfs[609]
# remap each part
images2 = np.zeros((n_parts, windowSize, windowSize, 3), dtype=int)
n = 0
for im in images:
    newImage = reMap(im,local_cdfs[n])
    images2[n] = newImage
    # images2[n] = im.copy()
    # for i in range(im.shape[0]):
    #     for j in range(im.shape[1]):
    #         images2[n][i][j] = local_cdfs[n][images2[n][i][j]]
    
    n +=1

# merge images
pil_L = Image.new('RGB', (image.shape[0],image.shape[1]),(250,250,250))
n = 0
factor = 1
for i in range(0,width,windowSize):
    for j in range(0,length,windowSize):
        pil_L.paste(Image.fromarray((images2[n] * factor).astype(np.uint8)) , (i,j))
        n +=1
L = np.array(pil_L,like=image)
plt.imshow(L)
plt.show()