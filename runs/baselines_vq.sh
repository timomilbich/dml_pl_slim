<<<<<<< Updated upstream
export GPU_TRAINING=2
=======
export GPU_TRAINING=5
>>>>>>> Stashed changes
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/VQ-DML/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"
export BETTER_EXCEPTIONS=1
export WANDB_KEY='8388187e7c47589ca2875e4007015c7536aede7f'
echo "WANDB_KEY: ${WANDB_KEY}"

### BASELINE CHECKS

### ... multisimilarity loss
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#               'model.params.config.Architecture.params.VQ=True' 'model.params.config.Architecture.params.n_e=500' 'model.params.config.Architecture.params.e_dim=1024' \
#               'model.params.config.Architecture.params.e_init=random_uniform' \
#                --savename baseline_bninception_vq500_UnifInit100_512_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'model.params.config.Architecture.params.VQ=True' 'model.params.config.Architecture.params.n_e=500' 'model.params.config.Architecture.params.e_dim=2048' \
               'model.params.config.Architecture.params.e_init=random_uniform' \
                --savename baseline_resnet50_vq500_UnifInit100_512_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

### ... larger image resolution during training, plz change the batch size accordingly
# python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
#                'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#                'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#                'data.params.train.params.arch=resnet50_frozen_normalize_imSize300' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#                'model.params.config.Architecture.params.VQ=True' 'model.params.config.Architecture.params.n_e=500' 'model.params.config.Architecture.params.e_dim=2048' \
#                 --savename baseline_resnet50_vq500_512_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
