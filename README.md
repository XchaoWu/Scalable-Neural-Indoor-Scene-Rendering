# Scalable-Neural-Indoor-Scene-Rendering

![](./pics/teaser.png)

We propose a scalable neural scene reconstruction and rendering method to support distributed training and interactive rendering of large indoor scenes.




<img src="./pics/table.gif" height="250"/> <img src="./pics/light.gif" height="250"/> <img src="./pics/floor.gif" height="250"/>



## Requirements

+ **System**: Ubuntu 16.04 or 18.04
+ **GCC/G++**: 7.5.0 or higher
+ **GPU** : we implement our method on RTX 3090. 
+ **CUDA version**: 11.1 or higher
+ **python**: 3.8 

To install python packages, run:

```shell
pip install -r requirements.txt
```



## Training

coming soon ...



## Rendering

Our method can render image of resolution 1280 x 720 in 20 FPS. 

### Build for rendering

To build the rendering project:

```shell
cd rendering
bash build.sh
```



### Interactive rendering

<img src='./pics/viewer.png' width=500>

We have provided a demo for interactive rendering. 

You can download the necessary rendering data [here](www.baidu.com). 

Then, 



