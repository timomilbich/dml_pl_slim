export GPU_TRAINING=4,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/compvis-nfs/user/tmilbich/PycharmProjects/VQ-DML/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using ViT backbone

############################
# ... multisimilarity loss #
############################

python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_normalize' \
               'data.params.train.params.arch=vit_base_patch16_224_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_normalize' \
               --savename baseline_multisimilarity_vit_base_patch16_224_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_base_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_in21k_normalize' \
               --savename baseline_multisimilarity_vit_base_patch16_224_in21k_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               --savename baseline_multisimilarity_vit_small_patch16_224_384f_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               --savename baseline_multisimilarity_vit_small_patch16_224_in21k_384f_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

