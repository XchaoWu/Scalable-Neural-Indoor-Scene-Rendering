# Scalable-Neural-Indoor-Scene-Rendering

![](./pics/pipeline.png)

 ### [Project Page](https://xchaowu.github.io/) | [Video](https://www.youtube.com/watch?v=hn-IVZZ-_lM) | [Paper]()  

We propose a scalable neural scene reconstruction and rendering method to support distributed training and interactive rendering of large indoor scenes.



## Requirements

+ **System**: Ubuntu 16.04 or 18.04
+ **GCC/G++**: 7.5.0 
+ **GPU** : we implement our method on RTX 3090. 
+ **CUDA version**: 11.1 or higher
+ **python**: 3.8 

To install required python packages:

```shell
conda env create -f env.yaml
```

C dependencies: [cnpy](https://github.com/rogersce/cnpy), [tqdm](https://github.com/tqdm/tqdm.cpp), [tinyply](https://github.com/ddiakopoulos/tinyply), 

## Rendering

Our method can render image of resolution 1280 x 720 in 20 FPS. 

For interactive rendering, you should also install [imgui](https://github.com/ocornut/imgui), [glfw-3.3.6](https://github.com/glfw/glfw).

For TensorRT acceleration, please first follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html), then install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt#option-1---without-plugins).

### Build for rendering

To build the rendering project:

```shell
cd rendering
bash build.sh
```



### Interactive rendering

<img src='./pics/viewer.png' width=500 >

We have provided a demo for interactive rendering. 

You can download the necessary rendering data [here](https://drive.google.com/file/d/1K2YFchSkCOerWZQ2UZtXWFzTG6Ph1n72/view?usp=sharing). Unzip file:

```shell
unzip data.zip
```

Then, replace the ` scene path` and `cnn path` in `rendering/config/base.yaml` with `data/renderData.npz` and `data/cnn.pth` , run:

```shell
bash demo.sh
```

## Training

Set your own **python directory** and **dependency path** in `./Scalable-Neural-Indoor-Scene-Rendering/training/src/make.sh`.

Then, Compilation:

```shell
./Scalable-Neural-Indoor-Scene-Rendering/training/src$ bash make.sh
```



For training a tile, please run:

```shell
python train.py -c {config_file} -t {tileIdx} -g {gpu_idx}
```

For training a group of tiles, please first make a file `group.txt` as follows:

```txt
tileIdx1
tileIdx2 
...
tileIdxN 
```

Then, run:

```shell
python train.py -c {config_file} -ts group.txt -g {gpu_idx}
```



