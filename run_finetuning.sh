#!/bin/bash
#SBATCH --job-name=sam_finetuning
#SBATCH --chdir=/sc/home/iven.schlegelmilch/sam2_gorilla_finetuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=20
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH -p aisc 
#SBATCH --account=aisc 
#SBATCH --qos=aisc 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:iven.schlegelmilch
#SBATCH --output=logs/%x-%j.out    # %x = jobname, %j = jobid
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

srun --container-image=/sc/home/iven.schlegelmilch/ivenschlegelmilch+gorillawatch+1.2.1.sqsh \
     --container-workdir=/workspaces \
     --container-mounts=/sc/home/iven.schlegelmilch/sam2_gorilla_finetuning:/workspaces/sam2_gorilla_finetuning \
     --container-writable \
     bash -c "cd /workspaces/sam2_gorilla_finetuning && \
              /opt/conda/envs/research/bin/python training/train.py \
              -c /sam2/configs/sam2.1_training/sam2.1_hiera_b+_gorilla_finetune.yaml \
              --use-cluster 0 \
              --num-gpus \$SLURM_GPUS_ON_NODE \
              --num-nodes \$SLURM_JOB_NUM_NODES \
              --partition \$SLURM_JOB_PARTITION \
              --qos \$SLURM_JOB_QOS \
              --account \$SLURM_JOB_ACCOUNT"
