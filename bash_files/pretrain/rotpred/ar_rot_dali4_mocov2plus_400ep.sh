python main_pretrain.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_dir /data \
    --train_dir ImageNet100/train \
    --val_dir ImageNet100/val \
    --max_epochs 400 \
    --gpus 0,1 \
    --accelerator ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler cosine \
    --lr 0.3 \
    --dali \
    --num_crops 4 \
    --classifier_lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.4\
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --name mocov2plus-400ep \
    --project ar_rot_mocov2 \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint \
    --method ar_rot_mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 65536 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier \
    --do_moco False \
    --do_rotatation True \
    --gar_weight 0 \
    --use_entropy_gar True \
    --grr_weight 0 \
    --lar_weight 0.5 \
    --use_entropy_lar True \
    --dense_split 7 \
    --dense_feats_dim 128\
    --lrot_topk 25 \
    --lrr_weight 0 \
    --arev_weight 0 \
    --rrev_weight 0 \
    --use_entropy_arev True \