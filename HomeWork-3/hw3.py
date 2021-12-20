import cv2
import os
import numpy as np

# get SIFT Keypoints and Descriptors
def get_sift_keypoints(img):
    img = img.copy()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img,None)
    sift_img = cv2.drawKeypoints(gray_img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_img, kp, desc

def draw_matched_keypoints(src_kpts, dst_kpts, good, src_img, dst_img, top_match_count, matchesMask=None):
    if matchesMask is not None:
        shortlisted_good = [match for match, mask in zip(good, matchesMask) if mask]
        shortlisted_good = sorted(shortlisted_good, key=lambda m: m.distance)
    else:
        shortlisted_good = sorted(good, key=lambda m: m.distance)
    
    if top_match_count is not None:
        shortlisted_good = shortlisted_good[:top_match_count]

    print(f"Average distance: {np.mean([m.distance for m in shortlisted_good])}")

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            singlePointColor = None,
            flags = 2)
    return cv2.drawMatches(src_img,src_kpts,dst_img,dst_kpts,shortlisted_good,None,**draw_params)

def main():
    img_dir = './HW3_Data'
    op_dir = './HW3_Outputs/'

    img_list = os.listdir(img_dir)

    src_img_data = []
    dst_img_data = []

    for img_path in img_list:
        # Load the image
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)
        img = cv2.imread(img_name)
        
        # Generate SIFT keypoints
        sift_img, kp, desc = get_sift_keypoints(img)
        cv2.imwrite(op_dir + "SIFT_image." + img_id + ".jpg", sift_img)
        print(img_id)
        print(f"No. of SIFT keypoints detected: {len(kp)}")

        # Segregate images into 2 buckets for creating image pairs
        if img_id.startswith("dst"):
            dst_img_data.append((img_id, img, desc, kp))
        else:
            src_img_data.append((img_id, img, desc, kp))

    
    '''
    Brute-Force Matching with SIFT Descriptors and Ratio Test
    This time, we will use BFMatcher.knnMatch() to get k best matches. 
    In this example, we will take k=2 so that we can apply ratio test explained by D.Lowe in his paper.
    '''

    bf = cv2.BFMatcher()
    for src_img_id, src_img, src_desc, src_kpts in src_img_data:
        for dst_img_id, dst_img, dst_desc, dst_kpts in dst_img_data:

            # Generate 2 nearest neighbours matches
            matches = bf.knnMatch(src_desc , dst_desc , k=2)
            print(src_img_id, dst_img_id)
            print(f"No. of matches: {len(matches)}")
            good = []

            # Retain the matches as per Lowe's ratio test
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            print(f"No. of matches after Lowe's ratio test: {len(good)}")
            
            print("All BF kNN matches")
            allKnnImg = draw_matched_keypoints(src_kpts, dst_kpts, good, src_img, dst_img, None)

            print("Top 20 BF kNN matches")
            knnImg = draw_matched_keypoints(src_kpts, dst_kpts, good, src_img, dst_img, 20)

            cv2.imwrite(op_dir + src_img_id + "." + dst_img_id + ".All_BFKNN.jpg", allKnnImg)
            cv2.imwrite(op_dir + src_img_id + "." + dst_img_id + ".Top_20_BFKNN.jpg", knnImg)

            src_pts = np.float32([ src_kpts[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([ dst_kpts[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            '''
            good = good_without_list
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape
            d = 0
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

            '''



            # Find homography matrix and the selected inliers as per RANSAC
            M, matchesMask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            print(f"No. of inlier matches after RANSAC: {sum(matchesMask)}")
            print("Homography matrix")
            print(M)

            print("All inlier matches")
            allInliersImg = draw_matched_keypoints(src_kpts, dst_kpts, good, src_img, dst_img, None, matchesMask)

            print("Top 10 inlier matches")
            finalImg = draw_matched_keypoints(src_kpts, dst_kpts, good, src_img, dst_img, 10, matchesMask)

            cv2.imwrite(op_dir + src_img_id + "." + dst_img_id + ".All_Inlier.jpg", allInliersImg)
            cv2.imwrite(op_dir + src_img_id + "." + dst_img_id + ".Top_10_Inlier.jpg", finalImg)



if __name__ == "__main__":
    main()