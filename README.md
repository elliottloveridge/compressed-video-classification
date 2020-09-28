# University of Bath Research Project

## Methods of Deep Model Compression for Video Classification

PyTorch Implementation of Elliott Loveridge's master's thesis, including code for video classification and model compression.

## Requirements

* [PyTorch 1.0.1.post2](http://pytorch.org/)
* OpenCV
* FFmpeg, FFprobe
* Python 3
* Distiller
* Docker

## Pre-trained models

Pre-trained models can be downloaded from a Google Drive [here](https://drive.google.com/drive/folders/1k93wkBQSZYpBSM1sTqwD3RzU9t7dahcr?usp=sharing).

Tested models:
 - 3D MobileNetv2
 - 3D ResNet
 - 3D CSN

 MobileNetV2's complexity may be adjusted via a 'width_multiplier' arg, with 'model_depth' choices possible for both ResNet and CSN.

## Dataset Preparation

### UCF101

If using Bath University's OGG, data has been saved in the following dir:
```misc
/mnt/slow0/ucf101/data/
```

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python utils/ucf101_json.py annotation_dir_path
```

All experiments use annotation file 1, saved within this repo.

### Dataset Dir

Data is assumed to be stored under the following schema:
```misc
/data/
    ucf101-videos/
        jpg/
            'ucf101-class'/
    results/
        benchmark/
            DDMM/
```

## Docker

This project was maintained within Docker to ensure the correct installation of Distiller and other relevant packages. If running this code on Bath University's OGG Service, reference the relevant Docker image via the below examples of running a test. The Dockerfile has been provided should you wish to create a similar image yourself.

## Running the code

Model configurations are given as follows:

```misc

ResNet-18	 : --model resnet  --model_depth 18  --resnet_shortcut A
ResNet-50	 : --model resnet  --model_depth 50  --resnet_shortcut B
ResNet-101	 : --model resnet  --model_depth 101 --resnet_shortcut B
MobileNetV2-1.0x  : --model mobilenetv2  --width_mult 1.0
CSN-50  : --model csn --model_depth 50
```

Example code runs are saved in ogg-run.py, and it is assumed this is used to run tests. Make sure to specify all parameters required for a given run. Also, __make sure__ the bash script is runnable - use 'chmod +x ogg-run.sh' for a script.

An example run is given as follows:

- Docker Code:
```bash
docker run --rm --ipc=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -v "$(pwd)":/app -v "/mnt/slow0/ucf101/data":/data elliottloveridge/distiller /app/ogg-run.sh
```

* NVIDIA_VISIBLE_DEVICES defines which available GPU to use

Then, ogg-run.py contains the following examples;

- Training from scratch:
```bash
python /app/compressed-3d-cnn/main.py --root_path /data \
    --video_path ucf101_videos/jpg/ \
    --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
    --result_path results \
    --dataset ucf101 \
    --n_classes 101 \
    --batch_size 32  \
    --model mobilenetv2 \
    --width_mult 1.0 \
    --learning_rate 0.1 \
    --n_val_samples 1 \
    --n_epochs 20 \
    --test
```

- Evaluation:
```bash
python /app/compressed-3d-cnn/utils/video_accuracy.py --root_path /data \
    --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
    --dataset ucf101 \
    --result_path results
```

* Evaluation will create a folder for MMYY (see __Dataset Dir__ above) and store it within the relevant sub-folder (this would be 'benchmark' for the above example)

* Example runs for model-compression methods are saved within ogg-run.py, see opts.py for all required compression arguments

* The ```distiller``` folder within ```compressed-3d-cnn``` contains YAML files required by the compression scheduler

### Augmentations

There are several augmentation techniques available. Please check spatial_transforms.py and temporal_transforms.py for the details of the augmentation methods.

Note: "RandomHorizontalFlip" and "RandomCrop" were used for training of UCF101

### Calculating Video Accuracy

In order to calculate video accuracy, you should first run the models with '--test' mode in order to create 'val.json'. Then, you need to run the evaluation script, given as an example above

## Acknowledgement
I'd like to thank both Kensho Hara for releasing his [codebase](https://github.com/kenshohara/3D-ResNets-PyTorch), the people who [extended](https://github.com/okankop/Efficient-3DCNNs) this work, and the team working on [Distiller](https://github.com/NervanaSystems/distiller) who allowed for model compression to be implemented.
