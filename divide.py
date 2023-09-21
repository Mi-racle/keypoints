import json
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def get_key(dic, value):
    return list(dic.keys())[list(dic.values()).index(value)]


ROOT = FILE = Path(__file__).resolve().parents[0]

SET_ID = [0, 1, 2]
SET_PROPORTION = [.625, .25, .125]

image_path = ROOT / 'datasets/animal_pose/images'
image_size = [640, 640]

obj_paths = []

for obj_name in os.listdir(image_path):
    obj_path = image_path / obj_name
    obj_paths.append(obj_path)

TOTAL = len(obj_paths)

set_ids = random.choices(population=SET_ID, weights=SET_PROPORTION, k=TOTAL)

fin = open(ROOT / 'datasets/animal_pose/keypoints.json', 'r')
data = json.load(fin)
fin.close()

if not os.path.exists(ROOT / 'datasets/animal'):
    os.mkdir(ROOT / 'datasets/animal')

if not os.path.exists(ROOT / 'datasets/animal/train'):
    os.mkdir(ROOT / 'datasets/animal/train')

if not os.path.exists(ROOT / 'datasets/animal/train' / 'images'):
    os.mkdir(ROOT / 'datasets/animal/train' / 'images')

if not os.path.exists(ROOT / 'datasets/animal/train' / 'labels'):
    os.mkdir(ROOT / 'datasets/animal/train' / 'labels')

if not os.path.exists(ROOT / 'datasets/animal/valid'):
    os.mkdir(ROOT / 'datasets/animal/valid')

if not os.path.exists(ROOT / 'datasets/animal/valid' / 'images'):
    os.mkdir(ROOT / 'datasets/animal/valid' / 'images')

if not os.path.exists(ROOT / 'datasets/animal/valid' / 'labels'):
    os.mkdir(ROOT / 'datasets/animal/valid' / 'labels')

if not os.path.exists(ROOT / 'datasets/animal/test'):
    os.mkdir(ROOT / 'datasets/animal/test')

if not os.path.exists(ROOT / 'datasets/animal/test' / 'images'):
    os.mkdir(ROOT / 'datasets/animal/test' / 'images')

if not os.path.exists(ROOT / 'datasets/animal/test' / 'labels'):
    os.mkdir(ROOT / 'datasets/animal/test' / 'labels')

for i, obj_path in tqdm(enumerate(obj_paths), total=len(obj_paths)):

    output = {'version': '5.2.1', 'flags': {}}
    shapes = []

    image_id = int(get_key(data['images'], obj_path.name))

    for annotation in data['annotations']:

        if annotation['image_id'] == image_id:

            for keypoint in annotation['keypoints']:

                shape = {'label': 'keypoint'}
                points = [[keypoint[0], keypoint[1]]]
                shape['points'] = points
                shapes.append(shape)

            break

    output['shapes'] = shapes

    dst_train_path = ROOT / 'datasets/animal/train/images' / obj_path.name
    lbl_train_path = dst_train_path.parents[1] / 'labels' / (os.path.splitext(obj_path.name)[0] + '.json')

    dst_valid_path = ROOT / 'datasets/animal/valid/images' / obj_path.name
    lbl_valid_path = dst_valid_path.parents[1] / 'labels' / (os.path.splitext(obj_path.name)[0] + '.json')

    dst_test_path = ROOT / 'datasets/animal/test/images' / obj_path.name
    lbl_test_path = dst_test_path.parents[1] / 'labels' / (os.path.splitext(obj_path.name)[0] + '.json')

    if set_ids[i] == 0:

        shutil.copyfile(obj_path, dst_train_path)
        with open(lbl_train_path, 'w') as f:
            train_json = json.dumps(output, indent=2)
            f.write(train_json)
            f.close()

    elif set_ids[i] == 1:

        shutil.copyfile(obj_path, ROOT / dst_valid_path)
        with open(lbl_valid_path, 'w') as f:
            valid_json = json.dumps(output, indent=2)
            f.write(valid_json)
            f.close()

    else:

        shutil.copyfile(obj_path, ROOT / dst_test_path)
        with open(lbl_test_path, 'w') as f:
            test_json = json.dumps(output, indent=2)
            f.write(test_json)
            f.close()

