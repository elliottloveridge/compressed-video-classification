#! /bin/bash

## benchmark - mobilenetv2

# # ucf101-mobilnetv2 (inc. testing) - 50 epochs, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --batch_size 32  \
# --model mobilenetv2 \
# --width_mult 1.0 \
# --learning_rate 0.1 \
# --n_val_samples 1 \
# --n_epochs 50 \
# --checkpoint 10 \
# --test

## benchmark - csn

# # ucf101-csn (inc. testing) - 50 epochs, 50 depth, 0.1 learning rate, 10 checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data \
#   --video_path ucf101_videos/jpg/ \
#   --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
#   --result_path results \
#   --dataset ucf101 \
#   --n_classes 101 \
#   --batch_size 16  \
#   --model csn \
#   --model_depth 50 \
#   --learning_rate 0.1 \
#   --n_val_samples 1 \
#   --n_epochs 50 \
#   --checkpoint 10 \
#   --test

## benchmark - resnet18

# # ucf101-resnet18 (inc. testing) - 50 epochs, 50 depth, 0.1 learning rate, 10 checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data \
#   --video_path ucf101_videos/jpg/ \
#   --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
#   --result_path results \
#   --dataset ucf101 \
#   --n_classes 101 \
#   --batch_size 32  \
#   --model resnet \
#   --model_depth 18 \
#   --learning_rate 0.1 \
#   --n_val_samples 1 \
#   --n_epochs 50 \
#   --checkpoint 10 \
#   --test


## evaluation

# # ucf101 evaluation (after testing)
# python /app/compressed-3d-cnn/utils/video_accuracy.py --root_path /data \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --dataset ucf101 \
# --result_path results


## fine-tuning & transfer learning (pre-trained kinetics-600)

# # ucf101-mobilenetv2 (inc. testing) - 20 epochs, 0.01 learning rate, 5 checkpoint, kinetics pre-train
# python /app/compressed-3d-cnn/main.py --root_path /data \
#   --video_path ucf101_videos/jpg/ \
#   --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
#   --result_path results \
#   --pretrain_path /data/results/pretrain/kinetics_mobilenetv2_1.0x_RGB_16_best.pth \
#   --dataset ucf101 \
#   --n_classes 600 \
#   --n_finetune_classes 101 \
#   --ft_portion complete \
#   --model mobilenetv2 \
#   --width_mult 1.0 \
#   --learning_rate 0.01 \
#   --sample_duration 16 \
#   --batch_size 32 \
#   --checkpoint 5 \
#   --n_val_samples 1 \
#   --n_epochs 20 \
#   --test

# # ucf101-resnet-18 (inc. testing) - 20 epochs, 0.01 learning rate, 5 checkpoint, kinetics pre-train
# python /app/compressed-3d-cnn/main.py --root_path /data \
#   --video_path ucf101_videos/jpg/ \
#   --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
#   --result_path results \
#   --pretrain_path /data/results/pretrain/kinetics_resnet_18_RGB_16_best.pth \
#   --dataset ucf101 \
#   --n_classes 600 \
#   --n_finetune_classes 101 \
#   --ft_portion complete \
#   --model resnet \
#   --model_depth 18 \
#   --learning_rate 0.01 \
#   --sample_duration 16 \
#   --batch_size 32 \
#   --checkpoint 5 \
#   --downsample 1 \
#   --resnet_shortcut A \
#   --n_val_samples 1 \
#   --n_epochs 20 \
#   --test


## compression


## qat

# # ucf101-mobilenetv2-qat (inc. testing) - 20 epochs, 0.01 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --batch_size 32  \
# --model csn \
# --model_depth 50 \
# --learning_rate 0.1 \
# --n_val_samples 1 \
# --n_epochs 2 \
# --test \
# --compress \
# --compression_type qat \
# --compression_file /app/compressed-3d-cnn/distiller/linear-qat.yaml

# # ucf101-mobilenetv2-qat-fine-tuning (inc. testing) - 20 epochs, 0.01 learning_rate, 5 checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 600 \
# --pretrain_path /data/results/pretrain/kinetics_resnet_101_RGB_16_best.pth \
# --dataset ucf101 \
# --n_finetune_classes 101 \
# --ft_portion complete \
# --batch_size 16  \
# --model resnet \
# --model_depth 101 \
# --learning_rate 0.01 \
# --n_val_samples 1 \
# --n_epochs 20 \
# --checkpoint 5 \
# --test \
# --compress \
# --compression_type qat \
# --compression_file /app/compressed-3d-cnn/distiller/linear-qat.yaml


## ptq

# # ucf101-mobilenetv2-ptq (inc. testing)  1 epoch, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --batch_size 32  \
# --model mobilenetv2 \
# --width_mult 1.0 \
# --learning_rate 0.1 \
# --n_val_samples 1 \
# --n_epochs 1 \
# --test \
# --compress \
# --compression_type ptq \
# --compression_file /app/compressed-3d-cnn/distiller/linear-ptq.yaml


## element-wise pruning


# element-wise pruning of mobilenetv2 w/ last_layer fine-tuning from pre-trained model
python /app/compressed-3d-cnn/multi-compress.py --root_path /data \
  --video_path ucf101_videos/jpg/ \
  --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
  --result_path results \
  --dataset ucf101 \
  --n_classes 101 \
  --batch_size 32  \
  --model mobilenetv2 \
  --width_mult 1.0 \
  --learning_rate 0.01 \
  --n_val_samples 1 \
  --n_epochs 1 \
  --compress \
  --ft_portion last_layer \
  --pretrain_path results/benchmark/1009/ucf101_mobilenetv2_benchmark_20epochs_1009_best.pth \
  --summary_path /app/compressed-3d-cnn/model_summary/mobilenetv2.csv
  --test


## knowledge distillation

# # ucf101-resnet-101 to resnet-18 knowledge distillation training (inc. testing)
# python /app/compressed-3d-cnn/main.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --batch_size 32 \
# --model resnet \
# --model_depth 18 \
# --learning_rate 0.1 \
# --n_val_samples 1 \
# --n_epochs 2 \
# --compress \
# --compression_type kd \
# --t_model resnet \
# --t_model_depth 101 \
# --t_path /data/results/benchmark/1209/ucf101_resnet_benchmark_20epochs_1209_best.pth \
# --t_n_classes 600 \
# --kd_distill_wt 0.7 \
# --kd_student_wt 0.3 \
# --kd_temp 5.0 \
# --checkpoint 2 \
# --test


## other


# # distiller model summary
# python /app/compressed-3d-cnn/model-summary.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --model mobilenetv2 \
# --width_mult 1.0 \
# --resume_path results/benchmark/1009/ucf101_mobilenetv2_benchmark_20epochs_1009_best.pth


# # distiller pruning sensitivity analysis
# python /app/compressed-3d-cnn/model-sensitivity.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --resume_path results/benchmark/1109_1/ucf101_csn_benchmark_50epochs_1109_best.pth \
# --result_path results/sensitivity/ \
# --dataset ucf101 \
# --n_classes 101 \
# --batch_size 16  \
# --model csn \
# --model_depth 50 \
# --summary_path /app/compressed-3d-cnn/model_summary/csn50.csv
