import albumentations as al
import numpy as np


class Augmentor:
    def __init__(self, views: int = 1):

        self.views = views

        additional_targets = {}

        for i in range(1, self.views):

            additional_targets[f'image{i}'] = 'image'
            additional_targets[f'keypoints{i}'] = 'keypoints'

        self.transform = al.Compose(
            [
                al.RandomBrightnessContrast(p=0.6),
                al.GaussianBlur(p=0.6),
                al.Rotate(limit=60, p=0.5),
                al.Affine(p=0.6),
                al.HueSaturationValue(p=0.4)
            ],
            keypoint_params=al.KeypointParams(format='yx', remove_invisible=False),
            additional_targets=additional_targets
        )

    def __call__(self, image_list, keypoints_list, augment: int):

        assert self.views == len(image_list)

        transformed_image_lists = []
        transformed_keypoints_lists = []

        for i in range(augment):

            args = {
                'image': image_list[0],
                'keypoints': keypoints_list[0]
            }

            for j in range(1, self.views):

                args[f'image{j}'] = image_list[j]
                args[f'keypoints{j}'] = keypoints_list[j]

            transformed = self.transform(**args)

            transformed_image_list = [transformed['image']]
            transformed_keypoints_list = [transformed['keypoints']]

            for j in range(1, self.views):

                transformed_image_list.append(transformed[f'image{j}'])
                transformed_keypoints_list.append(transformed[f'keypoints{j}'])

            transformed_image_lists.append(transformed_image_list)
            transformed_keypoints_lists.append(transformed_keypoints_list)

        transformed_image_lists = np.array(transformed_image_lists)
        transformed_keypoints_lists = np.array(transformed_keypoints_lists)
        return transformed_image_lists, transformed_keypoints_lists


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
