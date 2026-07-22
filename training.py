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
        # Enforce strict 2D shapes: (Batch, 512)
        z_v = projected_video_embeddings.view(projected_video_embeddings.size(0), -1)
        z_a = target_audio_embeddings.view(target_audio_embeddings.size(0), -1)
        
        # Normalize embeddings to unit length for stable cosine contrastive learning
        z_v = F.normalize(z_v, p=2, dim=-1)
        z_a = F.normalize(z_a, p=2, dim=-1)
        
        # Safe matrix multiplication using .t() for 2D transpose
        logits = torch.matmul(z_v, z_a.t()) / self.temperature
        
        labels = torch.arange(logits.shape[0]).to(logits.device)
        
        loss_v2a = F.cross_entropy(logits, labels)
        loss_a2v = F.cross_entropy(logits.T, labels)
        
        return (loss_v2a + loss_a2v) / 2
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline execution on: {device}")
    
    # Paths pointing to Node-Local ultra fast scratch spaces
    job_id = os.environ.get("SLURM_JOB_ID", "local_test")
    
    TRAIN_FRAMES = f"/scratch/{job_id}/vggsound_train_frames"
    VAL_FRAMES = f"/scratch/{job_id}/vggsound_val_frames"
    TRAIN_AUDIO = f"/scratch/{job_id}/train_audio_embeddings"
    VAL_AUDIO = f"/scratch/{job_id}/val_audio_embeddings"
    
    # Initialize frozen HuggingFace Vision Backbone
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_vision_tower = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True).to(device)
    clip_vision_tower.eval()
    
    # Initialize Datasets and Loaders
    train_dataset = VGGSoundFrameDataset(TRAIN_FRAMES, TRAIN_AUDIO, processor)
    val_dataset = VGGSoundFrameDataset(VAL_FRAMES, VAL_AUDIO, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=safe_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=safe_collate_fn, num_workers=4, pin_memory=True)
    
    model = VideoAudioAttentionBridge(clip_dim=768).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
    
    num_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = ContrastiveLoss(temperature=0.07)
    
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        # -----------------------------------
        # TRAINING PHASE
        # -----------------------------------
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            if batch is None: continue
            
            pixel_values = batch["pixel_values"].to(device)
            audio_embeddings = batch["audio_embedding"].to(device)
            
            B, T, C, H, W = pixel_values.shape
            pixel_values_flat = pixel_values.view(B * T, C, H, W)
            
            chunk_size = 1000
            pooled_list = []
            
            with torch.no_grad():
                for i in range(0, pixel_values_flat.size(0), chunk_size):
                    chunk = pixel_values_flat[i : i + chunk_size]
                    out = clip_vision_tower(chunk)
                    pooled_list.append(out.pooler_output)
                
                frame_features = torch.cat(pooled_list, dim=0).view(B, T, -1)

            projected_video_embeddings = model(frame_features)
            loss = criterion(projected_video_embeddings, audio_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
        avg_train_loss = total_train_loss / max(1, train_batches)
        
        # -----------------------------------
        # VALIDATION PHASE
        # -----------------------------------
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if not batch: continue
                
                pixel_values = batch["pixel_values"].to(device)
                audio_embeddings = batch["audio_embedding"].to(device)
                
                B, T, C, H, W = pixel_values.shape
                pixel_values_flat = pixel_values.view(B * T, C, H, W)
                
                chunk_size = 1000
                pooled_list = []
                
                for i in range(0, pixel_values_flat.size(0), chunk_size):
                    chunk = pixel_values_flat[i : i + chunk_size]
                    out = clip_vision_tower(chunk)
                    pooled_list.append(out.pooler_output)
                
                frame_features = torch.cat(pooled_list, dim=0).view(B, T, -1)
                
                projected_video_embeddings = model(frame_features)
                val_loss = criterion(projected_video_embeddings, audio_embeddings)
                
                total_val_loss += val_loss.item()
                val_batches += 1
                
        avg_val_loss = total_val_loss / max(1, val_batches)
        
        # Step LR scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save regular epoch checkpoint
        torch.save(model.state_dict(), f"attention_bridge_epoch_{epoch+1}.pt")
        
        # Track and save the absolute best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_attention_bridge.pt")
            print(f" --> Best model saved with Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()