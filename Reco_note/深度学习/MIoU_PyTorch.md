### 1. 前言

- 前面的学习笔记记录了相关深度学习评价标准，详见：[链接](https://yearing1017.site/2020/02/07/语义分割指标MIoU/)
- 本文记录**PyTorch实现MIoU及Acc的计算**

### 2. 数据的读入

- 代码实现**基于已经预测完成的predict数据集以及之前的label数据集。**
- **MIoU及Acc等评价标准按道理应该在验证的步骤上进行计算。**

- 数据载入代码：

```python
import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import cv2
'''
transform = transforms.Compose([
    transforms.ToTensor()    
])
'''
class MIoUDataset(Dataset):

    def __init__(self, transform = None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('data/dataset1/train_anno_noaug'))

    def __getitem__(self, idx):
        # 读label
        label_name = os.listdir('data/dataset1/train_anno_noaug')[idx]
        label = cv2.imread('data/dataset1/train_anno_noaug/' + label_name,0)
        # 读 predict
        predict = cv2.imread('predict_train_noaug_gray/' + label_name,0)
        return label, predict

data = MIoUDataset()

# 数据加载时会调用 __getitem__内置方法
MIoU_dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=1)

if __name__ == "__main__":
    for index, (label, predict) in enumerate(MIoU_dataloader):
        print(index)
        print('=============')
        print(label)
        print('=============')
        #print(predict)
        print('=============')
        if index > 1:
            break

```

- 之前的代码将predict和label都使用了transforms.ToTensor()，该方法的定义如下：
  - 把一个取值范围是`[0,255]`的`PIL.Image`或者`shape`为`(H,W,C)`的`numpy.ndarray`，转换成形状为`[C,H,W]`，取值范围是`[0,1.0]`的`torch.FloatTensor`
- 由此可见，此方法改变了predict的数值，原本label的像素值都是0-3的数字，predict也是如此，但是使用了该方法之后，就会产生错误，于是，直接使用cv2进行读入numpy的array数据，像素值不变。

### 3. MIoU及相关指标计算

- **先计算混淆矩阵，再使用混淆矩阵进行计算相关指标。详见之前的文章。**

- 代码如下：

```python
import numpy as np
from MIoUData import MIoU_dataloader
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

if __name__ == "__main__":
    miou = Evaluator(4)
    miouVal = 0
    accVal = 0
    for index, (predict, label) in enumerate(MIoU_dataloader):
        predict = predict.cpu().numpy()
        label = label.cpu().numpy()
        miou.add_batch(label,predict)
        accVal += miou.Pixel_Accuracy()
        miouVal += miou.Mean_Intersection_over_Union()
        print('acc and miou are {},{}'.format(miou.Pixel_Accuracy(),miou.Mean_Intersection_over_Union()))
    print('all acc and miou are {},{}'.format(accVal/len(MIoU_dataloader),miouVal/len(MIoU_dataloader)))
```

- 运行部分图像结果：

> acc and miou are 0.9892818803484108,0.9319128304522656
> acc and miou are 0.9892833639005335,0.9319363323158244
> acc and miou are 0.9892969403417731,0.9319694031143316
> acc and miou are 0.9892938647223908,0.9318407203700755
> acc and miou are 0.9893197875680992,0.93184955893125
> acc and miou are 0.989345520314387,0.931858144092549
> acc and miou are 0.9893461325771837,0.9319528589609107
> acc and miou are 0.9893458439753606,0.9318839785743727
> acc and miou are 0.9893494149992506,0.9318666985378861
> acc and miou are 0.9892998829526765,0.9317006233067969
> acc and miou are 0.989281737115901,0.9316989501236608
> acc and miou are 0.9892693801153274,0.931834614092101
> acc and miou are 0.9892564381216598,0.932001647774716
> acc and miou are 0.9892662417154177,0.9320374170466159
> acc and miou are 0.9892896893007536,0.9320368448465783
> acc and miou are 0.9893149494674971,0.9320454759482171
> acc and miou are 0.9893400907628677,0.9320540593312194
> acc and miou are 0.9893325160926496,0.9322192313956732
> acc and miou are 0.9893406716554449,0.9322583181307156
> acc and miou are 0.9893655766282126,0.9322668243060588
> acc and miou are 0.9893903654938812,0.9322752839025673
> acc and miou are 0.9893818643259448,0.932216114863396
> acc and miou are 0.989363195149362,0.9322293385657303
> acc and miou are 0.9893878173828125,0.9322377286726266
> all acc and miou are 0.9878058596007321,0.9306004317843026

