# dog_breed_classification_and_retrival
A project aimed to classify dog breeds and search for the most matched image in the database

The detail experiments and usage can be found in my website:
https://bshoterj.github.io/2019/01/20/dog-retrival/
or my group member's github:
https://github.com/INFINITSY/dog_test

## Requirements

 - Python3  
 - PyTorch
 - keras
 - dataset: Stanford dog dataset (18 classes)
![class](https://github.com/BshoterJ/dog_breed_retrival/blob/master/screen_img/class.png)

## Demo

Golden retrieve(金毛）
![t1](https://github.com/BshoterJ/dog_breed_retrival/blob/master/screen_img/test1.jpg)

Siberian husky（哈士奇）
![t2](https://github.com/BshoterJ/dog_breed_retrival/blob/master/screen_img/test2.jpg)

Dobeman(杜宾）
![t3](https://github.com/BshoterJ/dog_breed_retrival/blob/master/screen_img/test3.jpg)
        
## Model

- object detection: YOLOv3
- image retrival: ResNet50 + Euclid Distance
![model](https://github.com/BshoterJ/dog_breed_retrival/blob/master/screen_img/QQ%E6%88%AA%E5%9B%BE20190118001637.png)
