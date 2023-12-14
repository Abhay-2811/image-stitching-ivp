import cv2
import numpy as np

def vertical_image_stitching(image1_path, image2_path):
    # Load images
    image2 = cv2.imread(image1_path)
    image1 = cv2.imread(image2_path)

    # Check if images are loaded successfully
    if image1 is None or image2 is None:
        print("Error loading images.")
        return

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    # FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find Homography
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp image2 onto image1
    result = cv2.warpPerspective(image2, M, (max(image1.shape[1], image2.shape[1]), image1.shape[0] + image2.shape[0]))
    result[0:image1.shape[0], :] = image1

    # Display result
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
vertical_image_stitching("IMG_4823.jpg", "IMG_4824.jpg")
