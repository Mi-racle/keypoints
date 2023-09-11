import albumentations as al


class Augmentor:
    def __init__(self):
        self.transform = al.Compose([
            al.HorizontalFlip(p=0.5),  # 随机水平翻转图像，概率为0.5
            al.RandomBrightnessContrast(p=0.2),  # 随机调整图像的亮度和对比度，概率为0.2
            al.GaussianBlur(p=0.2),  # 随机应用高斯模糊，概率为0.2
            al.Rotate(limit=30, p=0.5),  # 随机旋转图像，限制在±30度之间，概率为0.5
        ], keypoint_params=al.KeypointParams(format='xy'))

    def __call__(self, images, keypoints):
        transformed = self.transform(image=images, keypoints=keypoints)
        transformed_images = transformed['image']
        transformed_keypoints = transformed['keypoints']
        # TODO
