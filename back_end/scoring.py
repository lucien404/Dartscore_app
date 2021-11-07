import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from dart_location import back_location
MIN_MATCH_COUNT = 10



def perspective(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # print(len(kp1))
    # print(len(kp2))
    # print(des1.shape)
    # print(des2.shape)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    # 对应关键点连接图
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    return M, img3


def scoring(location, img, tempelete_img, tempelete_label):
    # 透视变换矩阵
    M, kpt_img = perspective(tempelete_img, img)

    # 图像透视变换
    imgOut = cv2.warpPerspective(img, M, (tempelete_img.shape[1], tempelete_img.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


    # 点位透视变换
    M = np.linalg.inv(M)

    scores = []
    for idx in range(location.shape[0]):
        location_out_x = (location[idx, 0] * M[0, 0] + location[idx, 1] * M[0, 1] + M[0, 2]*1) / (location[idx, 0] * M[2, 0] + location[idx, 1] * M[2, 1] + M[2, 2])
        location_out_y = (location[idx, 0] * M[1, 0] + location[idx, 1] * M[1, 1] + M[1, 2]*1) / (location[idx, 0] * M[2, 0] + location[idx, 1] * M[2, 1] + M[2, 2])
        #  透视变换  齐次坐标   3*3  3*1 = 3*1  2*1  x,y,z   x=x/z   y = y/z

        location_out_x = int(round(location_out_x))
        location_out_y = int(round(location_out_y))
        print(location_out_x,location_out_y)
        if location_out_x > 1024 or location_out_y > 1024:
            score = 0
        else:
            score = tempelete_label[location_out_y, location_out_x]
        # print('第{}次飞镖得分为：{}分！'.format(idx+1, score))
        scores.append(score)
    # print(tempelete_label.shape)
    return imgOut, scores, kpt_img



def main(filename='IMG_0506.JPG'):
    # 输入信息：模版图像，模版标签，新图像，飞镖点位
    tempelete_img = cv2.imread('x0.jpeg', 0)
    tempelete_label = cv2.imread('tempelete.png', 0)
    img_new = cv2.imread(filename, 0)
    img = Image.open(filename)
    # location = [[1500, 2000], [1500, 2100], [1262, 1249]]  # [列， 行]  如果输入为[行，列]要进行调换，否则会导致计算结果错误
    location = back_location(img)
    # print(location)
    location = np.array(location)
    # 图像缩放（加速配准）
    tempelete_img_resize = cv2.resize(tempelete_img, dsize=(1024, 1024))
    tempelete_label_resize = cv2.resize(tempelete_label, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
    img_new_resize = cv2.resize(img_new, dsize=(1024, 1024))
    location_resize = np.array([location[:, 0]/(img_new.shape[1]/img_new_resize.shape[1]), location[:, 1]/(img_new.shape[0]/img_new_resize.shape[0])]).transpose()
    # print(location)
    # print(location_resize)

    # 执行打分程序
    img_out, score, kpt_img = scoring(location_resize, img_new_resize, tempelete_img_resize, tempelete_label_resize)


    # ## 获取校正之后的图像
    # img_out = cv2.resize(img_out, dsize=(tempelete_img.shape[1], tempelete_img.shape[0]))
    #
    # plt.suptitle("The {} times Score is {} respectively!".format(len(score), str(score)[1:-1]))
    # plt.subplot(221)
    # plt.imshow(tempelete_img)
    # plt.subplot(222)
    # plt.imshow(img_out)
    # plt.subplot(223)
    # plt.imshow(img_new)
    # plt.subplot(224)
    # plt.imshow(kpt_img)
    # plt.show()
    print(score)
    score_string = str(score)
    return score_string

if __name__ == "__main__":
    main()