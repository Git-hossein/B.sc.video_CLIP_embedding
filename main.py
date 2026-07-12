import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPProcessor
from pipeline_utils import VGGSoundFrameDataset, safe_collate_fn, VideoAudioAttentionBridge
import os


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, projected_video_embeddings, target_audio_embeddings):
        # Calculate the similarity matrix grid between all videos and all audios in the batch
        # Resulting shape: (batch_size, batch_size)
        logits = torch.matmul(projected_video_embeddings, target_audio_embeddings.T) / self.temperature
        
        # Ground truth targets are along the main diagonal (index matching itself)
        labels = torch.arange(logits.shape[0]).to(logits.device)
        
        # Cross entropy minimizes distance for true pairs and maximizes for negatives
        loss_v2a = F.cross_entropy(logits, labels)
        loss_a2v = F.cross_entropy(logits.T, labels)
        
        return (loss_v2a + loss_a2v) / 2
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline execution on: {device}")
    
    # Paths pointing to Node-Local ultra fast scratch spaces
    job_id = os.environ.get("SLURM_JOB_ID", "local_test")
    
    # Paths pointing to Node-Local ultra fast scratch spaces
    TRAIN_FRAMES = f"/scratch/{job_id}/vggsound_train_frames"
    VAL_FRAMES = f"/scratch/{job_id}/vggsound_val_frames"
    TRAIN_AUDIO = f"/scratch/{job_id}/vggsound_train_audio"
    VAL_AUDIO = f"/scratch/{job_id}/vggsound_val_audio"
    
    # Initialize frozen HuggingFace Vision Backbone
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_vision_tower = CLIPVisionModel.from_pretrained(clip_model_name).to(device)
    clip_vision_tower.eval()
    
    # Initialize Datasets and Loaders
    train_dataset = VGGSoundFrameDataset(TRAIN_FRAMES, TRAIN_AUDIO, processor)
    val_dataset = VGGSoundFrameDataset(VAL_FRAMES, VAL_AUDIO, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=safe_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=safe_collate_fn, num_workers=4)
    
    model = VideoAudioAttentionBridge().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = ContrastiveLoss()
    
    for epoch in range(10):
        # -----------------------------------
        # TRAINING PHASE
        # -----------------------------------
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            if batch is None: continue
            
            pixel_values = batch["pixel_values"].to(device)
            audio_embeddings = batch["audio_embedding"].to(device)
            
            B, T, C, H, W = pixel_values.shape
            pixel_values_flat = pixel_values.view(B * T, C, H, W)
            
            with torch.no_grad():
                vision_outputs = clip_vision_tower(pixel_values_flat)
                frame_features = vision_outputs.pooler_output.view(B, T, -1)
                
            projected_video_embeddings = model(frame_features)
            loss = criterion(projected_video_embeddings, audio_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # -----------------------------------
        # VALIDATION PHASE (Fixed & Added!)
        # -----------------------------------
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                
                pixel_values = batch["pixel_values"].to(device)
                audio_embeddings = batch["audio_embedding"].to(device)
                
                B, T, C, H, W = pixel_values.shape
                pixel_values_flat = pixel_values.view(B * T, C, H, W)
                
                vision_outputs = clip_vision_tower(pixel_values_flat)
                frame_features = vision_outputs.pooler_output.view(B, T, -1)
                
                projected_video_embeddings = model(frame_features)
                val_loss = criterion(projected_video_embeddings, audio_embeddings)
                
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoints safely
        torch.save(model.state_dict(), f"attention_bridge_epoch_{epoch+1}.pt")



if __name__ == "__main__":
    main()