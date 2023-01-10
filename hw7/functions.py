import cv2
import numpy as np

def BruteForceMatcher(src_desc,test_desc,threshold=0.75):
    bf = cv2.BFMatcher()
    # crossCheck=True
    matches = bf.knnMatch(src_desc,test_desc, k=2)
    # matches = bf.match(src_desc,test_desc)

    good_fPoints = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good_fPoints.append([m])

    return good_fPoints

# TODO: flann matching method
# def flannMatcher(src_desc,test_desc):

def howSimilar(_kp1,_kp2,good_feature_points):
    number_keypoints = min(len(_kp1),len(_kp2))

    return ((len(good_feature_points) / number_keypoints) * 100)