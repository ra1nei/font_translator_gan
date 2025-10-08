python train.py \
  --dataroot /kaggle/working/font_translator_gan/datasets/my_data/train \
  --model font_translator_gan \
  --name train \
  --no_dropout \
  --gpu_ids 0 \
  --use_wandb \
  --wandb_project font_translator_gan \
  --wandb_run_name train