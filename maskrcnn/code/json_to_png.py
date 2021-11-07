
# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import torch
import os
import numpy as np
import torch
from PIL import Image
 
 
 
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
 
 
class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
 
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = np.array(masks,dtype=np.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        #torch.Size([2, 4]) torch.Size([2]) torch.Size([2, 536, 559]) torch.Size([1]) torch.Size([2]) torch.Size([2])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
 
    def __len__(self):
        return len(self.imgs)
import cv2
from skimage.transform import resize
for js in os.listdir(r'C:\Users\Administrator\Desktop\paper\j'):#1, 2, 4, 5
    name = js.replace('json','jpg').replace('_','.')
    # print(name)
    for data in os.listdir(os.path.join(r'C:\Users\Administrator\Desktop\paper\j',js)):
        # try:
            if data == 'label.png':
                mask = Image.open(os.path.join(r'C:\Users\Administrator\Desktop\paper\j',js,data))
                # mask = Image.open('label.png')
                w,h = mask.size
                paper = np.zeros(shape=(h, w))
                mask = np.array(mask)
                obj_ids = np.unique(mask)
                obj_ids = obj_ids[1:]
                masks = mask == obj_ids[:, None, None]
                pig = np.zeros(shape=(h, w))
                # paper_mask = masks[1]
                pig_mask = masks[0]
                for i in range(h):
                    for j in range(w):
                        if pig_mask[i][j] != 0:
                            pig[i][j] = 255
                        # if paper_mask[i][j] == 1:
                        #     paper[i][j] = 255
                cv2.imwrite(r'C:\Users\Administrator\Desktop\paper\label\{}'.format(name), pig)
                # cv2.imwrite(r'D:\MASKpicture\datas\7\json\papers\{}'.format(name), paper)
            elif data == 'img.png':
                image = cv2.imread(os.path.join(r'C:\Users\Administrator\Desktop\paper\j', js, data))
