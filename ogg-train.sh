#! /bin/bash
python /app/efficient-3d-cnn/main.py --root_path ~/ --video_path /mnt/slow0/ucf101/toy/train --annotation_path /app/efficient-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 2  --model mobilenetv2 --width_mult 1.0 \
