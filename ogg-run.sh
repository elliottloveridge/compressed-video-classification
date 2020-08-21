#! /bin/bash


## benchmark

# # ucf101-mobilnetv2 (inc. testing) - 1 epochs, 0.1 learning rate, no checkpoint
# python /app/quantization-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/quantization-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 1 --test


## compression

## qat

# # ucf101-mobilenetv2-qat (inc. testing) - 20 epochs, 0.1 learning rate, no checkpoint
# python /app/quantization-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/quantization-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 20 --test --compress --compression_type qat --compression_file /app/quantization-3d-cnn/distiller/linear-qat.yaml

## filter pruning

# # ucf101-mobilenetv2-fp (no testing) - 1 epoch, 0.1 learning rate, no checkpoint
# python /app/quantization-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/quantization-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 1 --compress --compression_type fp --compression_file /app/quantization-3d-cnn/distiller/fp-mobilenetv2.yaml


## other

# # DEBUG: not tested
# # distiller model summary
# python /app/quantization-3d-cnn/model-summary.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/quantization-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --model mobilenetv2 --width_mult 1.0

# DEBUG: not finished
# distiller pruning sensitivity analysis
python /app/distiller/model-sensitivity.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/quantization-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --n_val_samples 1 --n_epochs 1 --resume_path results/benchmark/1108/ucf101_mobilenetv2_50epochs_32batch-size_train-1108_best.pth
