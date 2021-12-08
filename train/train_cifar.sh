
DATASET=cifar10
PORT=10020

export DATA_DIR="/atlas/u/a7b23/data"
export CHECKPOINT_DIR=${PWD}"/cifar10_ckpts"

echo ${CHECKPOINT_DIR}

GPUS=1
LR=0.001
let BS=GPUS*32
T=0.07
LOSS_TYPE=cpc
MOCO_DIM=2048
EPOCHS=1000
recon_loss_weight=17500

echo "Starting AE + Contrastive Learning training"

python train.py --seed 0 --lr ${LR} --moco-dim ${MOCO_DIM} --dataset ${DATASET} --optimizer adamw \
--dist-url tcp://localhost:${PORT} --multiprocessing-distributed --moco-t ${T} --world-size 1 --rank 0 \
--mlp -j 4 --loss ${LOSS_TYPE} --epochs ${EPOCHS} --batch-size ${BS} \
--moco-dim ${MOCO_DIM} --aug-plus --cos --save-dir ${CHECKPOINT_DIR} --data ${DATA_DIR} \
--num_channels_enc 128 --num_channels_dec 128 --num_postprocess_cells 2 --num_preprocess_cells 2 \
--num_latent_scales 1 --num_latent_per_group 10 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
--num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 6 \
--use_se --res_dist --recon_loss_weight ${recon_loss_weight}

resume_path=${CHECKPOINT_DIR}"/checkpoint_recent.pth.tar"
latent_path=${CHECKPOINT_DIR}"/train_feats_cifar10.npy"

echo "Generating features"

python save_latents.py --seed 0 --moco-dim ${MOCO_DIM} --dataset ${DATASET} \
--dist-url tcp://localhost:${PORT} --multiprocessing-distributed --moco-t ${T} --world-size 1 --rank 0 \
-j 4 --batch-size ${BS} \
--moco-dim ${MOCO_DIM} --data ${DATA_DIR} \
--num_channels_enc 128 --num_channels_dec 128 --num_postprocess_cells 2 --num_preprocess_cells 2 \
--num_latent_scales 1 --num_latent_per_group 10 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
--num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 6 \
--use_se --res_dist --resume ${resume_path} --eval_mode "save_latents" --out_dir ${CHECKPOINT_DIR} \
--out_fname ${latent_path}

echo "Starting DDIM training"

cd ../d2c/diffusion/

CONFIG='cifar10_moco.yml'
EXP_PATH='cifar10_model'


python main.py --train_fname ${latent_path} --config ${CONFIG} --exp ${EXP_PATH} --doc run_0 --ni
