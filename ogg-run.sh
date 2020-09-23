#! /bin/bash


## benchmark - mobilenetv2

# # ucf101-mobilnetv2 (inc. testing) - 2 epochs, 0.1 learning rate, no checkpoint
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
# --n_epochs 2 \
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


## fine-tuning (pre-trained)

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
# --model mobilenetv2 \
# --width_mult 1.0 \
# --learning_rate 0.01 \
# --n_val_samples 1 \
# --n_epochs 2 \
# --test \
# --compress \
# --compression_type qat \
# --compression_file /app/compressed-3d-cnn/distiller/linear-qat.yaml

# ucf101-mobilenetv2-qat-fine-tuning (inc. testing) - 20 epochs, 0.01 learning_rate, 5 checkpoint
python /app/compressed-3d-cnn/main.py --root_path /data \
--video_path ucf101_videos/jpg/ \
--annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
--result_path results \
--dataset ucf101 \
--n_classes 600 \
--pretrain_path /data/results/pretrain/kinetics_resnet_101_RGB_16_best.pth \
--dataset ucf101 \
--n_finetune_classes 101 \
--ft_portion complete \
--batch_size 32  \
--model resnet \
--model_depth 101 \
--learning_rate 0.01 \
--n_val_samples 1 \
--n_epochs 20 \
--checkpoint 5 \
--test \
--compress \
--compression_type qat \
--compression_file /app/compressed-3d-cnn/distiller/linear-qat.yaml


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

# # ucf101-mobilenetv2-ep (inc. testing) - 1 epoch, 16 batch_size, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data \
#   --video_path ucf101_videos/jpg/ \
#   --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
#   --result_path results \
#   --dataset ucf101 \
#   --n_classes 101 \
#   --batch_size 16  \
#   --model mobilenetv2 \
#   --width_mult 1.0 \
#   --learning_rate 0.1 \
#   --n_val_samples 1 \
#   --n_epochs 5 \
#   --compress \
#   --compression_type ep \
#   --compression_file /app/compressed-3d-cnn/distiller/ep-mobilenetv2-test1.yaml \
#   --test


## knowledge distillation

# # ucf101-resnet-101 to resnet-18 knowledge distillation training (inc. testing)
# # NOTE: t_n_classes set as 101 as using fine-tuned teacher model
# python /app/compressed-3d-cnn/distillation.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --batch_size 16 \
# --model resnet \
# --model_depth 18 \
# --learning_rate 0.1 \
# --n_val_samples 1 \
# --n_epochs 5 \
# --compress \
# --compression_type kd \
# --t_model resnet \
# --t_model_depth 101 \
# --t_path /data/results/benchmark/1209/ucf101_resnet_benchmark_20epochs_1209_best.pth \
# --t_n_classes 101 \
# --kd_distill_wt 0.7 \
# --kd_student_wt 0.3 \
# --kd_temp 5.0 \
# --test

# # ucf101-resnet-101 to MobileNetV2 knowledge distillation training (inc. testing)
# # NOTE: t_n_classes set as 101 as using fine-tuned teacher model
# python /app/compressed-3d-cnn/distillation.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --batch_size 16 \
# --model mobilenetv2 \
# --learning_rate 0.1 \
# --n_val_samples 1 \
# --n_epochs 5 \
# --compress \
# --compression_type kd \
# --t_model resnet \
# --t_model_depth 101 \
# --t_path /data/results/benchmark/1209/ucf101_resnet_benchmark_20epochs_1209_best.pth \
# --t_n_classes 101 \
# --kd_distill_wt 0.7 \
# --kd_student_wt 0.3 \
# --kd_temp 5.0 \
# --test


## other


# # distiller model summary - uses resume_path
# python /app/compressed-3d-cnn/model-summary.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --model mobilenetv2 \
# --width_mult 1.0 \
# --resume_path results/fp/2908/ucf101_mobilenetv2_fp_1epochs_2908_best.pth


# # distiller pruning sensitivity analysis
# python /app/compressed-3d-cnn/model-sensitivity.py --root_path /data \
# --video_path ucf101_videos/jpg/ \
# --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
# --resume_path results/benchmark/1108/ucf101_mobilenetv2_50epochs_32batch-size_train-1108_best.pth \
# --result_path results \
# --dataset ucf101 \
# --n_classes 101 \
# --batch_size 32  \
# --model mobilenetv2 \
# --width_mult 1.0 \
# --n_val_samples 1 \
# --n_epochs 1


# # distiller new pruning file
# python /app/compressed-3d-cnn/new-prune.py --root_path /data \
#   --video_path ucf101_videos/jpg/ \
#   --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
#   --result_path results \
#   --dataset ucf101 \
#   --n_classes 101 \
#   --batch_size 32  \
#   --model mobilenetv2 \
#   --width_mult 1.0 \
#   --learning_rate 0.1 \
#   --n_val_samples 1 \
#   --n_epochs 1 \
#   --pretrain_path results/benchmark/1108/ucf101_mobilenetv2_50epochs_32batch-size_train-1108_best.pth


# # distller pruning test only
# python /app/compressed-3d-cnn/new-prune.py --root_path /data \
#   --video_path ucf101_videos/jpg/ \
#   --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
#   --result_path results \
#   --dataset ucf101 \
#   --n_classes 101 \
#   --batch_size 32  \
#   --model mobilenetv2 \
#   --width_mult 1.0 \
#   --learning_rate 0.1 \
#   --n_val_samples 1 \
#   --n_epochs 1 \
#   --resume_path results/benchmark/1009/ucf101_mobilenetv2_benchmark_20epochs_1009_best.pth \
#   --no_train \
#   --no_val \
#   --test
