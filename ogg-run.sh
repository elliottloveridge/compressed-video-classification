#! /bin/bash


## benchmark

# # ucf101-mobilnetv2 (inc. testing) - 1 epochs, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 1 --test

## evaluation

# ucf101 evaluation (after testing)
python /app/compressed-3d-cnn/utils/video_accuracy.py --root_path /data --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --dataset ucf101 --result_path results


## compression

## qat

# # ucf101-mobilenetv2-qat (inc. testing) - 20 epochs, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 20 --test --compress --compression_type qat --compression_file /app/compressed-3d-cnn/distiller/linear-qat.yaml

## filter pruning

# # ucf101-mobilenetv2-fp (no testing) - 1 epoch, 16 batch_size, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 16  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 1 --compress --compression_type fp --compression_file /app/compressed-3d-cnn/distiller/fp-mobilenetv2.yaml


## other

# # distiller model summary - uses resume_path
# python /app/compressed-3d-cnn/model-summary.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --model mobilenetv2 --width_mult 1.0 --resume_path results/benchmark/1108/ucf101_mobilenetv2_50epochs_32batch-size_train-1108_best.pth

# # distiller pruning sensitivity analysis
# python /app/compressed-3d-cnn/model-sensitivity.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --n_val_samples 1 --n_epochs 1 --resume_path results/benchmark/1108/ucf101_mobilenetv2_50epochs_32batch-size_train-1108_best.pth
