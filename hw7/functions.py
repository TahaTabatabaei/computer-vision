import cv2
import numpy as np
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
plt.style.use(plt.style.available[5])

def BruteForceMatcher(src_desc,test_desc,threshold=0.75):
    """
    Apply a brute-force method to find matching interest points between the source and test description vectors.
    Using a threshold, discard weak points that are far enough from each other. 
    
    NOTE: the threshold can be set, due to usecase. 0.75 is recommended. Lowe recommends a threshold
    in range [0.7, 0.8].

    Inputs:
        - src_desc: description vector of src image
        - test_desc: description vector of test image
        - threshold: limit bound on distance of 2 points, to be discarded

    Returns:
        Feature(interest) points which are good enough
        NOTE: we return points in two diffrent lists, due to diffrent use cases. 
    """
    bf = cv2.BFMatcher()
    #TODO: apply crossCheck=True
    
    # knn algorithm for matching. store 2 nearest points as 'k=2'
    matches = bf.knnMatch(src_desc,test_desc, k=2)

    good_fPoints = []
    matches1to2 = []
    # apply threshold
    for m in matches:
        # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < threshold * m[1].distance:
            good_fPoints.append(m[0])
            matches1to2.append([m[0]])

    return good_fPoints , matches1to2

# TODO: flann matching method
# def flannMatcher(src_desc,test_desc):

def howSimilar(_kp1,_kp2,good_feature_points):
    """
    Calculate the similarity percentage between the discovered key points, based on how many 'good common feature points' they have.

    Inputs:
        - _kp1: set of source image key points.
        - _kp2: set of test image key points.
        - good_feature_points: good commen feature points. Output of matcher function.

    Returns:
        Similarity percentage
    """
    number_keypoints = min(len(_kp1),len(_kp2))

    return ((len(good_feature_points) / number_keypoints) * 100)

def homography(pairs):
    """
    Homography, is a transformation that is occurring between two planes.In other words,
    it is a mapping between two planar projections of an image. It is represented by
    a 3x3 transformation matrix in a homogenous coordinates space.

    This function, calculates the matrix.

    Inputs:
        - pairs: 4 pairs of matches found in previous steps. Since the Homography matrix
        has 8 degrees of freedom, we need at least four pairs of corresponding points
        to solve for the values of the Homography matrix.

    Returns:
        Homography matrix.
    

    Read the following links for a complete discussion about homography
    transformation(highly recommended). The second link explaines instruction logic of 'row1' 
    and 'row2' in details. 
    https://mattmaulion.medium.com/homography-transform-image-processing-eddbcb8e4ff7
    https://towardsdatascience.com/understanding-homography-a-k-a-perspective-transformation-cacaed5ca17
    """
    rows = []
    for i in range(pairs.shape[0]):
        # select a pair of points
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)

        # multiplication matrix A
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)

    # considering the equation A = H b , Since we want the H matrix, we should 
    # use Singular Value Decomposition
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H


def random_point(matches, k=4):
    """
    Returns 'k' pairs of matched points randomly.

    Inputs:
        - matches: list of matching points.
        - k: number of desired random points. 

    Returns:
        Random pairs of points.
    """
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)


def get_error(points, H):
    """
    Estimate points using the given H matrix, and calculate the difference between the estimated
    points and the real points, which were used to build the H matrix in the previous step.

    Inputs:
        - points: matched pairs.
        - H: homography matrix.

    Returns:
        Calculated error.
    """
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors


def ransac(matches, threshold, iters):
    """
    Random sample consensus(RANSAC), is an iterative method for estimating a mathematical model
    from a data set that contains outliers. The RANSAC algorithm works by identifying the outliers
    in a data set and estimating the desired model using data that does not contain outliers. It also
    can be interpreted as an outlier detection method. 
    In this function, we repeatedly compute an H matrix, then evaluate it. At the end, the function
    returns the best possible H matrix and match points.

    Inputs:
        - matches: list of matching points.
        - threshold: error threshold.
        - iters: number of algorithm iterations.

    Returns:
        best match points and H matrix.


    NOTE: read following links for better understanding of algorithm:
    https://en.wikipedia.org/wiki/Random_sample_consensus
    https://www.mathworks.com/discovery/ransac.html#:~:text=Random%20sample%20consensus%2C%20or%20RANSAC,that%20does%20not%20contain%20outliers.
    """
    num_best_inliers = 0
    best_H = any
    best_inliers = any
    
    # repeat this steps for a prescribed number of iterations
    for i in range(iters):
        # Randomly select a set of matching points
        points = random_point(matches)

        # compute homography matrix
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        # outlier detection
        # evaluate the H matrix
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        # save better inliers and H matrix
        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    # print number of good inliers
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H


def stitch(left, right, H):
    """
    Stitch provided images.

    Inputs:
        - left: The image we assume as the main image placed on the left side.
        - right: the image to be wraped.
        - H: homography matrix.

    Returns:
        An stitched image.
    """
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image


def harris_corner_detector(img, window_size=5, k=0.04, threshold=0.4):
    # copy the image in an new array
    # img = np.array(img)

    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)

    height = img.shape[0]  # shape[0] outputs height.
    

    width = img.shape[1] # shape[1] outputs width.
    matrix_R = np.zeros((height, width))

    # Step 1 - Calculate the x e y image derivatives (dx e dy)
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)
    # dy, dx = np.gradient(gray)

    #   Step 2 - Calculate product and second derivatives (dx2, dy2 e dxy)
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx*dy

    offset = int(window_size / 2)
    #   Step 3 - Calcular a soma dos produtos das derivadas para cada pixel (Sx2, Sy2 e Sxy)

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sx2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])

            #   Step 4 - Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]
            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])

            #   Step 5 - Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R = det-k*(tr**2)
            matrix_R[y-offset, x-offset] = R

    #   Step 6 - Apply a threshold
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            value = matrix_R[y, x]
            if value > threshold:
                # cornerList.append([x, y, value])
                cv2.circle(img, (x, y), 3, (0, 255, 0))

    return img


