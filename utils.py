import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_heatmap(width, height, x, save_name):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        # 8*8网格
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 1e-6)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        # 函数applycolormap产生伪彩色图像
        # COLORMAP_JET模式，就常被用于生成我们所常见的 热力图
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(save_name, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))
