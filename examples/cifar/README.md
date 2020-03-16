# The complete implementation of the CIFAR10 dataset in the ResNet paper

The official implementation of ImageNet is included, but the parameters of this dataset cannot be extended to cifar10 because it brings more of the appearance of overfitting of neural networks.If you categorize cifar10 directly with the official implementation, the accuracy of the paper is far from being achieved.To this end, I implemented the following code based on the description in the [original paper](http://xxx.itp.ac.cn/abs/1512.03385).

Details about the models are below: 

|     *Method*      |*# Params*|*Error (paper).*|*Error (ours).*|
|:-----------------:|:--------:|:--------------:|:-------------:|
|    `resnet20`     |  0.27M   |    8.75%       |     7.99%     |
|    `resnet32`     |  0.46M   |    7.51%       |     7.35%     |
|    `resnet44`     |  0.66M   |    7.17%       |     6.70%     |
|    `resnet56`     |  0.85M   |    6.97%       |     6.60%     |
|    `resnet110`    |   1.7M   |6.43%(6.61±0.16)|     6.26%     |
|    `resnet1202`   |  19.4M   |    7.93%       |     6.18%     |
 

### Train
```text
python main.py data -a resnet20 --gpu 0 
```

### Evaluate
```text
python main.py data -a resnet20 --gpu 0 -e --resume model_best.pth
```

### download pre-trained weights

**from github release**

- [resnet20.pth](https://github.com/Lornatang/ResNet-PyTorch/releases/download/1.0/resnet20-081ffb5e.pth)
- [resnet32.pth](https://github.com/Lornatang/ResNet-PyTorch/releases/download/1.0/resnet32-b9948351.pth)
- [resnet44.pth](https://github.com/Lornatang/ResNet-PyTorch/releases/download/1.0/resnet44-f74dd615.pth)
- [resnet56.pth](https://github.com/Lornatang/ResNet-PyTorch/releases/download/1.0/resnet56-68aecbac.pth)
- [resnet110.pth](https://github.com/Lornatang/ResNet-PyTorch/releases/download/1.0/resnet110-000407b3.pth)
- [resnet1202.pth](https://github.com/Lornatang/ResNet-PyTorch/releases/download/1.0/resnet1202-f3b1deed.pth)

**from Baidu cloud disk**

- 
