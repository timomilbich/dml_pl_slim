export GPU_TRAINING=0
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/compvis-nfs/user/tmilbich/PycharmProjects/VQ-DML/experiments/training_models'
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

### ... usage of random_uniform intialization for codebook initialization with resnet50
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'model.params.config.Architecture.params.VQ=True' 'model.params.config.Architecture.params.n_e=500' 'model.params.config.Architecture.params.e_dim=2048' \
#               'model.params.config.Architecture.params.e_init=random_uniform' \
#                --savename baseline_resnet50_vq500_UnifInit100_512_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

### ... larger image resolution during training, plz change the batch size accordingly
# python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
#                'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#                'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#                'data.params.train.params.arch=resnet50_frozen_normalize_imSize300' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#                'model.params.config.Architecture.params.VQ=True' 'model.params.config.Architecture.params.n_e=500' 'model.params.config.Architecture.params.e_dim=2048' \
#                 --savename baseline_resnet50_vq500_512_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

### ... usage of feature_clustering for codebook initialization with resnet50
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'model.params.config.Architecture.params.VQ=True' 'model.params.config.Architecture.params.n_e=2000' 'model.params.config.Architecture.params.e_dim=2048' \
#               'model.params.config.Architecture.params.e_init=feature_clustering' 'model.params.config.Architecture.params.k_e=64' \
#                --savename baseline_resnet50_vq2000_k64_ClusterInit_512_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=70' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenAll_normalize' \
#               'data.params.train.params.arch=resnet50_frozenAll_normalize' 'data.params.validation.params.arch=resnet50_frozenAll_normalize' \
#               'model.params.config.Architecture.params.VQ=VQ_multihead' 'model.params.config.Architecture.params.n_e=2000' \
#               'model.params.config.Architecture.params.e_init=feature_clustering' 'model.params.config.Architecture.params.k_e=32' \
#               'model.params.config.Architecture.params.block_to_quantize=3' \
#               'lightning.trainer.precision=16' \
#                --savename vqMH_16bit_resnet50_vq2000_block3_k32_ClusterInit_512_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=70' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
#               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
#               'model.params.config.Architecture.params.VQ=VQ_multihead' 'model.params.config.Architecture.params.n_e=2000' \
#               'model.params.config.Architecture.params.e_init=feature_clustering' 'model.params.config.Architecture.params.k_e=32' \
#               'model.params.config.Architecture.params.block_to_quantize=3' \
#               'lightning.trainer.precision=16' \
#                --savename vqMH_16bit_resnet50_removeBN4.1_2_vq2000_block3_k32_ClusterInit_512_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.resnet50_transformer.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenAll_1x1conv_normalize' \
#               'data.params.train.params.arch=resnet50_frozenAll_1x1conv_normalize' 'data.params.validation.params.arch=resnet50_frozenAll_1x1conv_normalize' \
#               'model.params.config.Architecture.params.VQ=VQ_multihead' 'model.params.config.Architecture.params.n_e=4000' 'model.params.config.Architecture.params.e_dim=512' \
#               'model.params.config.Architecture.params.e_init=feature_clustering' 'model.params.config.Architecture.params.k_e=4' \
#               'lightning.trainer.precision=16' 'data.params.batch_size=112'\
#                --savename vqMH_transf_16bit_resnet50_reduce512_vq4000_k4_ClusterInit_128_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=vq_resnet_transformer' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.resnet50_transformer.Network' 'model.params.config.Architecture.params.arch=resnet50_1x1conv_frozenAll_normalize' \
               'data.params.train.params.arch=resnet50_1x1conv_frozenAll_normalize' 'data.params.validation.params.arch=resnet50_1x1conv_frozenAll_normalize' \
               'model.params.config.Architecture.params.VQ=VQ_multihead' 'model.params.config.Architecture.params.n_e=2000' 'model.params.config.Architecture.params.e_dim=256' \
               'model.params.config.Architecture.params.e_init=feature_clustering' 'model.params.config.Architecture.params.k_e=2' 'model.params.config.Architecture.params.beta=2'\
               'lightning.trainer.precision=16' 'data.params.batch_size=112' 'model.warmup_iterations=50'\
                --savename vqMH_transf_w50_16bit_resnet50_vq2000_k2_ClusterInit_256_LegacyBeta2_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml