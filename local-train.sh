#! /bin/bash
python3 /app/efficient-3d-cnn/main.py --video_path ~/video-classification/toy/train --annotation_path ~/Documents/GitHub/research-project/efficient-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --model mobilenetv2 --width_mult 1.0
