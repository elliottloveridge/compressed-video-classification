#! /bin/bash


## benchmark - mobilenetv2

# # ucf101-mobilnetv2 (inc. testing) - 1 epochs, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 1 --test


## benchmark - csn

# # ucf101-csn (inc. testing) - 1 epochs, 50 depth, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model csn --model_depth 50 --learning_rate 0.1 --n_val_samples 1 --n_epochs 1 --test


## evaluation

# # ucf101 evaluation (after testing)
# python /app/compressed-3d-cnn/utils/video_accuracy.py --root_path /data --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --dataset ucf101 --result_path results


## pre-training

# # ucf101-mobilenetv2 (inc. testing) - 1 epochs, 0.001 learning rate, no checkpoint, kinetics pre-train
python /app/compressed-3d-cnn/main.py --root_path /data \
  --video_path ucf101_videos/jpg/ \
  --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json \
  --result_path results \
  --pretrain_path /data/results/pretrain/kinetics_mobilenetv2_1.0x_RGB_16_best.pth \
  --dataset ucf101 \
  --n_classes 600 \
  --n_finetune_classes 101 \
  --ft_portion complete \
  --model mobilenetv2 \
  --width_mult 1.0 \
  --learning_rate 0.001 \
  --sample_duration 16 \
  --batch_size 32 \
  --checkpoint 1 \
  --n_val_samples 1 \
  --test


## compression


## qat

# # ucf101-mobilenetv2-qat (inc. testing) - 20 epochs, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 20 --test --compress --compression_type qat --compression_file /app/compressed-3d-cnn/distiller/linear-qat.yaml


## filter pruning
# FIXME: does not work - change to element wise pruning

# # ucf101-mobilenetv2-fp (inc. testing) - 1 epoch, 16 batch_size, 0.1 learning rate, no checkpoint
# python /app/compressed-3d-cnn/main.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 16  --model mobilenetv2 --width_mult 1.0 --learning_rate 0.1 --n_val_samples 1 --n_epochs 1 --test --compress --compression_type fp --compression_file /app/compressed-3d-cnn/distiller/fp-mobilenetv2.yaml


## knowledge distillation

# # ucf101-resnet-101 to resnet-18 knowledge distillation training (inc. testing)
# # FIXME: t_n_classes set as 600, is this right?
# python /app/compressed-3d-cnn/distillation.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 16 --model resnet --model_depth 18 --learning_rate 0.001 --n_val_samples 1 --n_epochs 1 --compress --compression_type kd --t_model resnet --t_model_depth 101 --t_path /data/results/pretrain/kinetics_resnet_101_RGB_16_best.pth --t_n_classes 600 --kd_distill_wt 0.7 --kd_student_wt 0.3 --kd_temp 5.0 --test



## other


# # distiller model summary - uses resume_path
# python /app/compressed-3d-cnn/model-summary.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --model mobilenetv2 --width_mult 1.0 --resume_path results/fp/2908/ucf101_mobilenetv2_fp_1epochs_2908_best.pth


# # distiller pruning sensitivity analysis
# python /app/compressed-3d-cnn/model-sensitivity.py --root_path /data --video_path ucf101_videos/jpg/ --annotation_path /app/compressed-3d-cnn/annotation_UCF101/ucf101_01.json --result_path results --dataset ucf101 --n_classes 101 --batch_size 32  --model mobilenetv2 --width_mult 1.0 --n_val_samples 1 --n_epochs 1 --resume_path results/benchmark/1108/ucf101_mobilenetv2_50epochs_32batch-size_train-1108_best.pth
