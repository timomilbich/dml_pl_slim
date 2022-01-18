export GPU_TRAINING=3 # if this is given in command line, add quotes as "2,5,6,"
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/home/zwu/dml_pl_slim/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"
export BETTER_EXCEPTIONS=1

### BASELINE CHECKS
# ... margin loss
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#              --savename baseline_marginloss_cub200_no_distance_mining --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#              --savename baseline_marginloss_cub200_bninception --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

## ... multisimilarity loss
python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
               'model.params.config.Vectorquantizer.params.n_e=250' \
              --savename baseline_multisimilarity_cub200_250 --debug --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
               'model.params.config.Vectorquantizer.params.n_e=500' \
              --savename baseline_multisimilarity_cub200_500 --debug --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
               'model.params.config.Vectorquantizer.params.n_e=750' \
              --savename baseline_multisimilarity_cub200_750 --debug --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
               'model.params.config.Vectorquantizer.params.n_e=1500' \
              --savename baseline_multisimilarity_cub200_1500 --debug --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
               'model.params.config.Vectorquantizer.params.n_e=2000' \
              --savename baseline_multisimilarity_cub200_2000 --debug --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#              --savename baseline_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#             --savename baseline_multisimilarity_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml


##  ... proxyanchor loss

#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#              --savename baseline_proxyanchor_cub200 --debug --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
