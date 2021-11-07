import torch
from torchvision.transforms import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def pltshow(im):
    plt.figure(figsize=(15,6))
    if im.ndim == 3:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(im, 'gray')
    plt.show()

def back_location(img):

    model = torch.load('350.pkl', map_location='cpu')
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # img = Image.open('IMG_0506.JPG')
    all_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        ])
    img = all_transforms(img)

    with torch.no_grad():
        prediction = model([img])
    # Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    location_res = []
    for itemn in range(len(prediction[0]['masks'])):
        if prediction[0]['scores'][itemn].item() < 0.8:
            print(prediction[0]['scores'][itemn].item())
            continue
        imgarray = prediction[0]['masks'][itemn,0].mul(255).byte().numpy()
        # imori = img.mul(255).permute(1, 2, 0).byte().numpy()#
        img_mask = Image.fromarray(imgarray)
        threshold = 50
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)
        img_mask.point(table, '1')
        img = imgarray
        _, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY )
        # pltshow(img)
        # img.shape
        from cv2 import imshow as cv2_imshow

        # from google.colab.patches import cv2_imshow
        # print (cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(contours)
        # cnt = contours[1]
        # a = cv2.drawContours(imori, [cnt], 0, (0,255,0), 3)
        # cv2_imshow(cv2.UMat.get(a))

        from matplotlib import pyplot as plt
        countour_len = len(contours)
        location = list()
        for index in range(countour_len):
            mu=cv2.moments(contours[index],False)
            x,y=[mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]

            cnt = contours[index]
            hull = cv2.convexHull(cnt)
            maxx,maxy=0,0
            maxdis = 0
            for i in hull:
            # print (i[0])
                if ((i[0][0] - x)**2 + (i[0][1] - y)**2) > maxdis:
                    maxdis = ((i[0][0] - x)**2 + (i[0][1] - y)**2)
                    maxx = i[0][0]
                    maxy = i[0][1]
            location.append([maxx,maxy])
            # print(maxx,maxy)
            # location[i][1] = maxx
            # location[i][2] = maxy
            # result = cv2.circle(imori,(maxx,maxy),20,(0,0,255),-1)
            location_res.append(location[0])
    print(location_res)
    return location_res

def pltshow(im):
    plt.figure(figsize=(15,6))
    if im.ndim == 3:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(im, 'gray')
    plt.show()