import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPProcessor
from tqdm import tqdm

from pipeline_utils import VGGSoundFrameDataset, safe_collate_fn, VideoAudioAttentionBridge

def extract_and_save(loader, clip_vision_tower, model, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # tqdm gives us a clean extraction counter in the logs
        for batch in tqdm(loader, desc=f"Extracting to {os.path.basename(output_dir)}", mininterval=20.0, maxinterval=60.0):
            if not batch: continue
            
            video_ids = batch["video_id"]
            pixel_values = batch["pixel_values"].to(device)
            
            B, T, C, H, W = pixel_values.shape
            pixel_values_flat = pixel_values.view(B * T, C, H, W)
            
            # 1. Extract raw features from the frozen CLIP Vision Model
            vision_outputs = clip_vision_tower(pixel_values_flat)
            frame_features = vision_outputs.pooler_output.view(B, T, -1)
            
            # 2. Pass through our best Epoch 3 Attention Bridge
            projected_video_embeddings = model(frame_features) # Shape: (B, 512)
            
            # Move back to CPU and convert to numpy array format
            embeddings_np = projected_video_embeddings.cpu().numpy()
            
            # 3. Save each video embedding individually using its unique YouTube ID
            for i, video_id in enumerate(video_ids):
                save_path = os.path.join(output_dir, f"{video_id}.npy")
                np.save(save_path, embeddings_np[i])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running embedding extraction engine on: {device}")
    
    job_id = os.environ.get("SLURM_JOB_ID", "local_test")
    
    # Fast scratch locations
    TRAIN_FRAMES = f"/scratch/{job_id}/vggsound_train_frames"
    VAL_FRAMES = f"/scratch/{job_id}/vggsound_val_frames"
    TRAIN_AUDIO = f"/scratch/{job_id}/train_audio_embeddings"
    VAL_AUDIO = f"/scratch/{job_id}/val_audio_embeddings"
    
    # Target permanent storage paths (saved safely back in your home workspace)
    # This ensures they persist long after the /scratch folder disappears!
    OUTPUT_TRAIN_DIR = "/home/sherkat/singularity_videoembedding/train_attention_video_embeddings"
    OUTPUT_VAL_DIR = "/home/sherkat/singularity_videoembedding/val_attention_video_embeddings"
    
    # Initialize frozen CLIP components
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_vision_tower = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True).to(device)
    clip_vision_tower.eval()
    
    # Initialize Datasets and Loaders
    train_dataset = VGGSoundFrameDataset(TRAIN_FRAMES, TRAIN_AUDIO, processor)
    val_dataset = VGGSoundFrameDataset(VAL_FRAMES, VAL_AUDIO, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=safe_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=safe_collate_fn, num_workers=4)
    
    # Instantiate architecture with your mandatory clip_dim=768 fix
    model = VideoAudioAttentionBridge(clip_dim=768).to(device)
    
    # Load the specific Epoch 03 checkpoint
    checkpoint_path = "/home/sherkat/singularity_videoembedding/attention_bridge_epoch_3.pt"
    print(f"Loading weights from optimal checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # Set model explicitly to evaluation mode (locks dropout)
    
    # Run extraction phases
    print("Starting Training set extraction...")
    extract_and_save(train_loader, clip_vision_tower, model, device, OUTPUT_TRAIN_DIR)
    
    print("Starting Validation set extraction...")
    extract_and_save(val_loader, clip_vision_tower, model, device, OUTPUT_VAL_DIR)
    
    print("All video embeddings successfully extracted and saved to home workspace!")

if __name__ == "__main__":
    main()