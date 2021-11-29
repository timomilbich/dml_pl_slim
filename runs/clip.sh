export GPU_TRAINING=3,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using CLIP backbone

# ... margin loss
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=clip' 'lightning.trainer.max_epochs=60' \
#      --savename baseline_marginloss_clip512f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'lightning.logger.params.group=clip' 'lightning.trainer.max_epochs=60' \
#        --savename baseline_marginloss_clip128_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=clip' 'lightning.trainer.max_epochs=60' \
#      --savename baseline_marginloss_clipS512_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=clip' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=84' \
      --savename baseline_marginloss_clipR50128_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml


# ... multisimilarity loss
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=clip' 'lightning.trainer.max_epochs=60' \
#              --savename baseline_multisimilarity_clip512f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'lightning.logger.params.group=clip' 'lightning.trainer.max_epochs=60' \
#              --savename baseline_multisimilarity_clip128_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=clip' 'lightning.trainer.max_epochs=60' \
#              --savename baseline_multisimilarity_clip512_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=clip' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=84' \
              --savename baseline_multisimilarity_clipR50128_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml