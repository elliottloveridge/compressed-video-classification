#! /bin/bash
python /app/efficient-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/efficient-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --model mobilenetv2 --test --no_train --no_val --resume_path results/ucf101_mobilenetv2_50epochs_32batch-size_train-1108_best.pth
