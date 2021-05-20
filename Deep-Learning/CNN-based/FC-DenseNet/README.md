Semantic Segmentation Practice
===

## Data Set
**Cambridge-Driving Labeled Video Database**
> [The original dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid)

> [From Kaggle](https://www.kaggle.com/carlolepelaars/camvid)

![](https://i.imgur.com/2mbfLH9.png)



## Network Architecture
Pytorch implementation of FC-DenseNet([paper](https://arxiv.org/pdf/1611.09326.pdf))

![](https://i.imgur.com/jQDOU6D.png)



## How to use
```
  python main.py 
```
**Option**
- --img_w : width of image 
- --img_h : height of image
- --device: cpu or cuda
- --seed  : initial seed 
- --growth_rate: the size of feature map is number of layer times growth rate
## Requirement
- matplotlib  3.3.3
- numpy       1.19.4
- pandas      1.1.4
- tensorflow  2.4.0
- tensorboard 2.1
- tqdm        4.55.1
- python3     3.8.5
- pytorch     1.7.1+cu110
