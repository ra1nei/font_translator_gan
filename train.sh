python train.py \
  --dataroot ./datasets/my_data/train \
  --model font_translator_gan \
  --name train \
  --no_dropout \
  --gpu_ids 0 \
  --use_wandb \
  --wandb_project font_translator_gan \
  --wandb_run_name font_translator_gan_train \
  --batch_size 128