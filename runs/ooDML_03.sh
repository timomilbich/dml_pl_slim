export GPU_TRAINING=0
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/compvis-nfs/user/tmilbich/PycharmProjects/VQ-DML/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"
export BETTER_EXCEPTIONS=1
export WANDB_KEY='8388187e7c47589ca2875e4007015c7536aede7f'
echo "WANDB_KEY: ${WANDB_KEY}"

### DML runs based on ooDML splits

############################
# ... multisimilarity loss #
############################

#####################################################################################
python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=ooDML_non_vq_02' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=1' 'data.params.validation.params.ooDML_split_id=1' \
               'model.params.config.Architecture.params.VQ=False' \
               --savename baseline_frozen_removeBN_1_2_normalize_512_multisimilarity_sop_s1 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=ooDML_non_vq_02' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=2' 'data.params.validation.params.ooDML_split_id=2' \
               'model.params.config.Architecture.params.VQ=False' \
               --savename baseline_frozen_removeBN_1_2_normalize_512_multisimilarity_sop_s2 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=ooDML_non_vq_02' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=3' 'data.params.validation.params.ooDML_split_id=3' \
               'model.params.config.Architecture.params.VQ=False' \
               --savename baseline_frozen_removeBN_1_2_normalize_512_multisimilarity_sop_s3 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=ooDML_non_vq_02' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=4' 'data.params.validation.params.ooDML_split_id=4' \
               'model.params.config.Architecture.params.VQ=False' \
               --savename baseline_frozen_removeBN_1_2_normalize_512_multisimilarity_sop_s4 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=ooDML_non_vq_02' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=5' 'data.params.validation.params.ooDML_split_id=5' \
               'model.params.config.Architecture.params.VQ=False' \
               --savename baseline_frozen_removeBN_1_2_normalize_512_multisimilarity_sop_s5 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=ooDML_non_vq_02' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=6' 'data.params.validation.params.ooDML_split_id=6' \
               'model.params.config.Architecture.params.VQ=False' \
               --savename baseline_frozen_removeBN_1_2_normalize_512_multisimilarity_sop_s6 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=ooDML_non_vq_02' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=7' 'data.params.validation.params.ooDML_split_id=7' \
               'model.params.config.Architecture.params.VQ=False' \
               --savename baseline_frozen_removeBN_1_2_normalize_512_multisimilarity_sop_s7 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=ooDML_non_vq_02' "lightning.logger.params.wandb_key=${WANDB_KEY}" 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.params.arch=resnet50_frozenPart_normalize' 'data.params.validation.params.arch=resnet50_frozenPart_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=8' 'data.params.validation.params.ooDML_split_id=8' \
               'model.params.config.Architecture.params.VQ=False' \
               --savename baseline_frozen_removeBN_1_2_normalize_512_multisimilarity_sop_s8 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml