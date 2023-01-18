import cv2
import numpy as np

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

def stitch(matches,keypoints_l,keypoints_r,left_img,right_img):
    matches = sorted(matches, key=lambda x:x.distance)

    # Map each match to corresponding keypoints
    points = matches[:4]
    # points = matches[:3]

    points = map(lambda m: (keypoints_l[m.queryIdx], keypoints_r[m.trainIdx]), points)

    # Map keypoints to coordinates
    pts_left = []
    pts_right = []
    for left_pt, right_pt in points:
        pts_left.append(list(left_pt.pt))
        pts_right.append(list(right_pt.pt))

    print(pts_left)
    print(pts_right)

    # Convert to numpy arrays
    pts_left = np.array(pts_left, dtype='float32')
    pts_right = np.array(pts_right, dtype='float32')

    # # Plot points on the images
    # colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255,255,0]]

    # right_img_pts = right_img.copy()
    # for i, (x, y) in enumerate(pts_right):
    #     x, y = int(x), int(y)
    #     right_img_pts[y-10:y+10, x-10:x+10, :] = colors[i]

    # left_img_pts = left_img.copy()
    # for i, (x, y) in enumerate(pts_left):
    #     x, y = int(x), int(y)
    #     left_img_pts[y-10:y+10, x-10:x+10, :] = colors[i]

    # # Create affine transformation matrix
    # M = cv2.getAffineTransform(pts_right, pts_left)
    # right_image_transformed = cv2.warpAffine(right_img, M, (right_img.shape[1], right_img.shape[0]))

    # # Create a mask
    # mask = np.all((left_img == 0), axis=2)
    # result = np.zeros_like(left_img)
    # result[mask] = 255


    # # Merge the images
    # full_mask = np.repeat(mask.reshape(mask.shape+(1,)), 3, axis=2)
    # result = left_img + (full_mask * right_image_transformed)

    return pts_left, pts_right



def homography(matches,ptsA,ptsB,reprojThresh=0.4):
    # computing a homography requires at least 4 matches
    # if len(matches) > 4:
        # construct the two sets of points
        # ptsA = np.float32([_kp1[i] for (_, i) in matches])
        # ptsB = np.float32([_kp2[i] for (i, _) in matches])

        # compute the homography between the two sets of points
    (h, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
        reprojThresh)
        
    # return the matches along with the homograpy matrix
    # and status of each matched point
    return (h, status)

    # otherwise, no homograpy could be computed
    # return None

def st(imageA,imageB,H):
    # apply a perspective warp to stitch the images together

    # (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H,
        (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    return result

# https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

