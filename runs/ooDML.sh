export GPU_TRAINING=0,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/compvis-nfs/user/tmilbich/PycharmProjects/VQ-DML/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"
export BETTER_EXCEPTIONS=1

### DML runs based on ooDML splits

############################
# ... multisimilarity loss #
############################

# baseline (Resnet50 Dino) - SOP
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=1' 'data.params.validation.params.ooDML_split_id=1' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s1 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=2' 'data.params.validation.params.ooDML_split_id=2' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s2 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=3' 'data.params.validation.params.ooDML_split_id=3' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s3 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=4' 'data.params.validation.params.ooDML_split_id=4' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s4 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=5' 'data.params.validation.params.ooDML_split_id=5' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s5 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=6' 'data.params.validation.params.ooDML_split_id=6' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s6 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=7' 'data.params.validation.params.ooDML_split_id=7' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s7 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=8' 'data.params.validation.params.ooDML_split_id=8' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s8 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=9' 'data.params.validation.params.ooDML_split_id=9' \
               --savename msloss_resnet50_dino_frozen_normalize_384_sop_s9 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml