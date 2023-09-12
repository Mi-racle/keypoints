import json
import os

import albumentations as al
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms


class Augmentor:
    def __init__(self):

        self.transform = al.Compose([
            al.HorizontalFlip(p=0.5),
            al.RandomBrightnessContrast(p=0.6),
            al.GaussianBlur(p=0.6),
            al.Rotate(limit=60, p=0.5),
            al.Affine(p=0.6),
            al.HueSaturationValue(p=0.4)
        ], keypoint_params=al.KeypointParams(format='yx', remove_invisible=False))

    def __call__(self, aimage, keypoints, multiple: int):

        transformed_images = []
        transformed_keypointses = []

        for i in range(multiple):

            transformed = self.transform(image=aimage, keypoints=keypoints)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']

            transformed_images.append(transformed_image)
            transformed_keypointses.append(transformed_keypoints)

        transformed_images = np.array(transformed_images)
        transformed_keypointses = np.array(transformed_keypointses)
        return transformed_images, transformed_keypointses
        # TODO


# img_path = 'datasets/testset2/images/Camera-GE501GC-C0A801E1-Snapshot-20230311-083624-820-820353307379_BMP.rf.b3113c05ac020e2b6e4d3abea5f6d96f.jpg'
# image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# lbl_path = 'datasets/testset2/labels/Camera-GE501GC-C0A801E1-Snapshot-20230311-083624-820-820353307379_BMP.rf.b3113c05ac020e2b6e4d3abea5f6d96f.json'
# f = open(lbl_path, 'r')
# dic = json.load(f)
# f.close()
# points = [(shape['points'][0][1], shape['points'][0][0]) for shape in dic['shapes']]
#
# augmentor = Augmentor()
# ai, aks = augmentor(image, points, 1)
#
# transformer = transforms.ToPILImage()
#
# ti = transformer(ai)
# drawer = ImageDraw.Draw(ti)
#
# for keypoint in aks:
#     y, x = keypoint[0], keypoint[1]
#     drawer.ellipse(((x - 5, y - 5), (x + 5, y + 5)), fill=(0, 255, 0))
#
# ti.save('image.jpg')

