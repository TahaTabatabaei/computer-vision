import cv2
import numpy as np
import random
from tqdm.notebook import tqdm

def BruteForceMatcher(src_desc,test_desc,threshold=0.75):
    """
    Apply a brute-force method to find matching interest points between the source and test description vectors.
    Using a threshold, discard weak points that are far enough from each other. 
    
    NOTE: the threshold can be set, due to usecase. 0.75 is recommended.

    Inputs:
        - src_desc: description vector of src image
        - test_desc: description vector of test image
        - threshold: limit bound on distance of 2 points, to be discarded

    Returns:
        feature(interest) points which are good enough
        
    """
    bf = cv2.BFMatcher()
    # bf = cv2.DescriptorMatcher_create("BruteForce")
    #TODO: apply crossCheck=True
    
    # knn algorithm for matching. store 2 nearest points as 'k=2'
    matches = bf.knnMatch(src_desc,test_desc, k=2)
    # matches = bf.match(src_desc,test_desc)

    # print(matches)

    good_fPoints = []
    matches1to2 = []
    mGoodTuple = []
    # apply threshold
    for m in matches:
        # if m.distance < threshold*n.distance:
        #     good_fPoints.append([m])
        # print(m)
        # print(len(m))
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < threshold * m[1].distance:
            mGoodTuple.append((m[0].trainIdx, m[0].queryIdx))
            good_fPoints.append(m[0])
            matches1to2.append([m[0]])


    return good_fPoints , matches1to2 , mGoodTuple

# TODO: flann matching method
# def flannMatcher(src_desc,test_desc):

def howSimilar(_kp1,_kp2,good_feature_points):
    """
    Calculate similarity percentage between image, based on how many 'good comment feature points' they have.

    Inputs:
        - _kp1: set of source image keypoints
        - _kp2: set of test image keypoints
        - good_feature_points: good commen feature points. Output of matcher function

    Returns:
        similarity percentage
    
    """
    number_keypoints = min(len(_kp1),len(_kp2))

    return ((len(good_feature_points) / number_keypoints) * 100)

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H


def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)


def get_error(points, H):
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
    num_best_inliers = 0
    best_H = any
    best_inliers = any
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H


def stitch_img(left, right, H):
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




# https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

