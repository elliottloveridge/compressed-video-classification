#! /bin/bash
python /app/efficient-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/efficient-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --n_val_samples 1 --checkpoint 5 --n_epochs 20
