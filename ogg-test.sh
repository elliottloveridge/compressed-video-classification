#! /bin/bash
python /app/efficient-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/efficient-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results/qat/0608 --dataset ucf101 --n_classes 101 --n_epochs 20 --model mobilenetv2 --test --no_train --no_val --resume_path results/qat/0608/ucf101_mobilenetv2_20epochs_16_qat-0608_best.pth
