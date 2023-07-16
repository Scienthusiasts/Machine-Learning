import os
from PIL import Image
"""
该脚本将图像灰度化 & resize
"""

# 新建文件夹
os.mkdir('./resized')

# 遍历文件夹中的所有图像文件
root = 'E:/DataSets(no used yet)/celeba_hq_256/'
for i, file in enumerate(os.listdir(root)):
    # 忽略非图像文件
    if not file.endswith('.jpg'):
        continue

    # 打开图像文件
    im = Image.open(os.path.join(root, file)).convert('L')

    # 调整图像大小为 128x128
    im_resized = im.resize((64, 64))
    print(i)

    # 保存调整后的图像到新文件夹中
    im_resized.save('./resized/'+str(i)+'.jpg')