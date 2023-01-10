import cv2
import numpy as np

def BruteForceMatcher(src_desc,test_desc,threshold=0.75):
    """
    Apply brute-force method to find matching interest points between src & test description vectors.
    Using threshold, we discard weak points (points that are far enough from eachother) 
    
    NOTE: the threshold can be set, due to usecase, 0.75 is recommended

    Inputs:
        - src_desc: description vector of src image
        - test_desc: description vector of test image
        - threshold: limit bound on distance of 2 points, to be discarded

    Returns:
        feature(interest) points which are good enough
        
    """
    bf = cv2.BFMatcher()
    #TODO: apply crossCheck=True
    
    # knn algorithm for matching. store 2 nearest points as 'k=2'
    matches = bf.knnMatch(src_desc,test_desc, k=2)

    # apply threshold
    good_fPoints = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good_fPoints.append([m])

    return good_fPoints

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