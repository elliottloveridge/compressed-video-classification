# University of Bath Research Project

### Methods of Deep Model Compression for Video Classification

PyTorch Implementation of the Elliott Loveridge's master's thesis, including code for video classification and model compression.

### Requirements

- Python 3
- PyTorch 1.0.1.post2
- FFmpeg, FFprobe
- Distiller

### Pretrained Models

All pretrained models can be downloaded from here

Implemented models:

3D SqueezeNet
3D MobileNet
3D ShuffleNet
3D MobileNetv2
3D ShuffleNetv2
For state-of-the-art comparison, the following models are also evaluated:

ResNet-18
ResNet-50
ResNet-101
ResNext-101
All models (except for SqueezeNet) are evaluated for 4 different complexity levels by adjusting their 'width_multiplier' with 2 different hardware platforms.

Results

Results of Efficient 3DCNNs

Dataset Preparation

Kinetics

Download videos using the official crawler.

Locate test set in video_directory/test.
Different from the other datasets, we did not extract frames from the videos. Insted, we read the frames directly from videos using OpenCV throughout the training. If you want to extract the frames for Kinetics dataset, please follow the preperation steps in Kensho Hara's codebase. You also need to modify the kinetics.py file in the datasets folder.

Generate annotation file in json format similar to ActivityNet using utils/kinetics_json.py

The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.
python utils/kinetics_json.py train_csv_path val_csv_path video_dataset_path dst_json_path
Jester

Download videos here.
Generate n_frames files using utils/n_frames_jester.py
python utils/n_frames_jester.py dataset_directory
Generate annotation file in json format similar to ActivityNet using utils/jester_json.py
annotation_dir_path includes classInd.txt, trainlist.txt, vallist.txt
python utils/jester_json.py annotation_dir_path
UCF-101

Download videos and train/test splits here.
Convert from avi to jpg files using utils/video_jpg_ucf101_hmdb51.py
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
Generate n_frames files using utils/n_frames_ucf101_hmdb51.py
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
Generate annotation file in json format similar to ActivityNet using utils/ucf101_json.py
annotation_dir_path includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt
python utils/ucf101_json.py annotation_dir_path
Running the code

Model configurations are given as follows:

ShuffleNetV1-1.0x : --model shufflenet   --width_mult 1.0 --groups 3
ShuffleNetV2-1.0x : --model shufflenetv2 --width_mult 1.0
MobileNetV1-1.0x  : --model mobilenet    --width_mult 1.0
MobileNetV2-1.0x  : --model mobilenetv2  --width_mult 1.0 
SqueezeNet      : --model squeezenet --version 1.1
ResNet-18      : --model resnet  --model_depth 18  --resnet_shortcut A
ResNet-50      : --model resnet  --model_depth 50  --resnet_shortcut B
ResNet-101      : --model resnet  --model_depth 101 --resnet_shortcut B
ResNeXt-101      : --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
Please check all the 'Resource efficient 3D CNN models' in models folder and run the code by providing the necessary parameters. An example run is given as follows:

Training from scratch:
python main.py --root_path ~/ \
    --video_path ~/datasets/jester \
    --annotation_path Efficient-3DCNNs/annotation_Jester/jester.json \
    --result_path Efficient-3DCNNs/results \
    --dataset jester \
    --n_classes 27 \
    --model mobilenet \
    --width_mult 0.5 \
    --train_crop random \
    --learning_rate 0.1 \
    --sample_duration 16 \
    --downsample 2 \
    --batch_size 64 \
    --n_threads 16 \
    --checkpoint 1 \
    --n_val_samples 1 \
Resuming training from a checkpoint:
python main.py --root_path ~/ \
    --video_path ~/datasets/jester \
    --annotation_path Efficient-3DCNNs/annotation_Jester/jester.json \
    --result_path Efficient-3DCNNs/results \
    --resume_path Efficient-3DCNNs/results/jester_shufflenet_0.5x_G3_RGB_16_best.pth \
    --dataset jester \
    --n_classes 27 \
    --model shufflenet \
    --groups 3 \
    --width_mult 0.5 \
    --train_crop random \
    --learning_rate 0.1 \
    --sample_duration 16 \
    --downsample 2 \
    --batch_size 64 \
    --n_threads 16 \
    --checkpoint 1 \
    --n_val_samples 1 \
Training from a pretrained model. Use '--ft_portion' and select 'complete' or 'last_layer' for the fine tuning:
python main.py --root_path ~/ \
    --video_path ~/datasets/jester \
    --annotation_path Efficient-3DCNNs/annotation_UCF101/ucf101_01.json \
    --result_path Efficient-3DCNNs/results \
    --pretrain_path Efficient-3DCNNs/results/kinetics_shufflenet_0.5x_G3_RGB_16_best.pth \
    --dataset ucf101 \
    --n_classes 600 \
    --n_finetune_classes 101 \
    --ft_portion last_layer \
    --model shufflenet \
    --groups 3 \
    --width_mult 0.5 \
    --train_crop random \
    --learning_rate 0.1 \
    --sample_duration 16 \
    --downsample 1 \
    --batch_size 64 \
    --n_threads 16 \
    --checkpoint 1 \
    --n_val_samples 1 \
Augmentations

There are several augmentation techniques available. Please check spatial_transforms.py and temporal_transforms.py for the details of the augmentation methods.

Note: Do not use "RandomHorizontalFlip" for trainings of Jester dataset, as it alters the class type of some classes (e.g. Swipe_Left --> RandomHorizontalFlip() --> Swipe_Right)

Calculating Video Accuracy

In order to calculate viceo accuracy, you should first run the models with '--test' mode in order to create 'val.json'. Then, you need to run 'video_accuracy.py' in utils folder to calculate video accuracies.

Calculating FLOPs

In order to calculate FLOPs, run the file 'calculate_FLOP.py'. You need to fist uncomment the desired model in the file.

Citation

Please cite the following article if you use this code or pre-trained models:

@article{kopuklu2019resource,
  title={Resource Efficient 3D Convolutional Neural Networks},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Kose, Neslihan and Gunduz, Ahmet and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:1904.02422},
  year={2019}
}
Acknowledgement

We thank Kensho Hara for releasing his codebase, which we build our work on top.
