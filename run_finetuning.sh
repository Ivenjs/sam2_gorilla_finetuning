#!/bin/bash
#SBATCH --job-name=gamma_sweep
#SBATCH --chdir=/sc/home/iven.schlegelmilch/sam2_gorilla_finetuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH -w gx08,gx09,gx11,gx12,gx13
#SBATCH -p aisc 
#SBATCH --account=aisc 
#SBATCH --qos=aisc 
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

srun --container-image=/sc/home/iven.schlegelmilch/ivenschlegelmilch+gorillawatch+1.2.1.sqsh \
     --container-workdir=/workspaces \
     --container-mounts=/sc/home/iven.schlegelmilch/sam2_gorilla_finetuning:/workspaces/sam2_gorilla_finetuning \
     --container-writable \
     bash -c "cd /workspaces/sam2_gorilla_finetuning && \
              /opt/conda/envs/research/bin/python training/train.py \
              -c /sam2/configs/sam2.1_training/sam2.1_hiera_b+_gorilla_finetune.yaml \
              --use-cluster 1 \
              --num-gpus \$SLURM_GPUS_ON_NODE \
              --num-nodes \$SLURM_JOB_NUM_NODES \
              --partition \$SLURM_JOB_PARTITION \
              --qos \$SLURM_JOB_QOS \
              --account \$SLURM_JOB_ACCOUNT"
