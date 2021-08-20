# RFSong-7993进行行人检测
重新设计的RFBNet300,模型参数量只有0.99MB，AP达到0.80(相比RFBNet300降了两个点)，速度却可以达到200FPS，并且在多个其他任务表现良好，例如钢筋检测，人手检测等等

文章链接：https://zhuanlan.zhihu.com/p/76491446

## 环境
容器为tomzhai_pytorch_gpu/python

在python3.6 Ubuntu16.04 pytorch1.1下进行了实验

- 编译NMS 

```
cd utils/
python build.py build_ext --inplace
```



# 训练自己的数据集

1. 新建一个文件夹VOCdevkit，将自己的VOC格式数据集的VOC2007文件夹移动到VOCdevkit中，例如我的就是/home/common/wangsong/VOCdevkit/VOC2007
```Shell
$VOCdevkit/
$VOCdevkit/VOC2007/
$VOCdevkit/VOC2007/Annotations/
$VOCdevkit/VOC2007/ImageSets/
$VOCdevkit/VOC2007/JPEGImages/
```

2. 修改data/config.py中的数据集路径 `VOCroot = '/home/common/wangsong/VOC/VOCdevkit' ` 改为自己数据集对应的路径

3. 修改针对VOC行人的代码部分，可以参考[我的博客](https://zhuanlan.zhihu.com/p/75086049 "悬停显示")查看修改的地方
  - train_RFB.py的trainsets进行改动：`train_sets = [('2007', 'trainval'))]`
  - data/voc0712.py和data/voc_eval.py这两个文件可以直接用[钢筋检测](https://zhuanlan.zhihu.com/p/75086049 "悬停显示")里面的替换

  **或者在群文件下载RFSong多数据集联合训练文件夹（因为多数据集联合训练就统一用的VOC2007格式）**

4. 运行train_RFB

# 测试

修改demo.py

```
parser.add_argument('-m', '--trained_model', default='weights/epoches_112.pth',
                    type=str, help='Trained state_dict file path to open')
mkdir images_out
cv2.imwrite('./images_out/' + img_name, image)
python demo.py
```



# 转onnx

```
python convert_to_onnx.py
```



# 与RFB不同的是
- 代码更方便进行自己设计网络
- 网络非常轻量级只有3.1 MB
- 网络速度相比RFB有显著提升,能够达到200FPS

# 更新
在VOC0712+COCO+PRW+SYSU总共大概十万张图片的数据集上进行了训练，**模型代码和权重都放在了群文件**，实测效果得到了显著提升，有需要的可以直接加群下载使用。

欢迎加群交流，云深不知处-目标检测 763679865

