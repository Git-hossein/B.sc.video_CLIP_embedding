#!/bin/bash
#SBATCH --job-name="vggsound_attention_train"
#SBATCH --partition=day
#SBATCH --time=12:00:00                     # Giving it 12 hours max (adjust as needed)
#SBATCH --gres=gpu:2080ti:1                 # Requesting 1 GPU (2080ti or A4000 work well)
#SBATCH --cpus-per-task=4                   # Matches our DataLoader num_workers=4
#SBATCH --mem=45G                           # 45GB RAM is perfect for processing image batches
#SBATCH --output=%x_%j.log       # Clean standard prints go here
#SBATCH --error=%x_%j.err        # All structural error dumps go here!


# Exit immediately if any command fails
set -e

echo "=== Job started at $(date) ==="
echo "=== Allocation ID: $SLURM_JOB_ID ==="

# 1. Create localized directory blocks inside the fast node-scratch array
echo "Creating fast local scratch directories..."
mkdir -p /scratch/$SLURM_JOB_ID/vggsound_train_frames
mkdir -p /scratch/$SLURM_JOB_ID/vggsound_val_frames

# 2. Extract the Audio Embeddings into Scratch
echo "Unpacking audio embeddings into scratch..."
cp -r ~/train_audio_embeddings /scratch/$SLURM_JOB_ID/
cp -r ~/val_audio_embeddings  /scratch/$SLURM_JOB_ID/ 

# 3. Extract the Image Frames into Scratch
echo "Unpacking video frame tarballs into scratch (this may take a few minutes)..."
tar -xf ~/vggsound_train_frames.tar -C /scratch/$SLURM_JOB_ID/vggsound_train_frames/ --strip-components=1
tar -xf ~/vggsound_val_frames.tar -C /scratch/$SLURM_JOB_ID/vggsound_val_frames/ --strip-components=1

echo "Data extraction complete! Verifying contents..."
echo "Train videos count: $(ls /scratch/$SLURM_JOB_ID/vggsound_train_frames | wc -l)"

# 4. Jump into the execution workspace
cd ~/singularity_videoembedding/

# 5. Launch our split Python training job inside your container
# Note: --nv enables GPU hardware acceleration inside Singularity
echo "Launching training loop execution inside Singularity container..."
singularity exec --nv vid_embedding_image.sif python3 B.sc.video_CLIP_embedding/main.py

echo "=== Job finished successfully at $(date) ==="