python train.py \
  --dataroot /kaggle/working/font_translator_gan/my_data \
  --model font_translator_gan \
  --name test_new_dataset \
  --no_dropout \
  --gpu_ids 0 \
  --use_wandb \
  --wandb_project font_translator_gan \
  --wandb_run_name test_new_dataset
