import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F 

class VGGSoundFrameDataset(Dataset):
    def __init__(self, frames_dir, audio_embeddings_dir, clip_processor, target_frames=50):
        """
        Args:
            frames_dir (str): Path to the unpacked frames directory (will be in /scratch)
            audio_embeddings_dir (str): Path to your pre-saved Wav2CLIP vectors
            clip_processor: The feature extractor/processor from HuggingFace CLIP
            target_frames (int): Exactly how many frames to feed the attention layer (5fps * 10s = 50)
        """
        self.frames_dir = frames_dir
        self.audio_dir = audio_embeddings_dir
        self.clip_processor = clip_processor
        self.target_frames = target_frames
        
        # Get list of video unique IDs (assuming folder names correspond to video IDs)
        self.video_ids = sorted(os.listdir(frames_dir))

    def __len__(self):
        return len(self.video_ids)

    def _load_and_sample_frames(self, video_id):
        """
        returns a list of RGB converted frames (= target_frames) belonging to a video or None if there are no frames for that video.
        
        :param self: Description
        :param video_id: Description
        """
        video_folder = os.path.join(self.frames_dir, video_id)
        # Get all frame images sorted alphabetically/numerically
        all_frames = sorted([f for f in os.listdir(video_folder) if f.endswith(('.jpg'))])
        
        num_frames = len(all_frames)
        if num_frames == 0:
            print(f"No frames found for video {video_id}")
            return None
            
        # Standardize to exactly target_frames (50) using uniform index sampling
        indices = torch.linspace(0, num_frames - 1, self.target_frames).long()
        sampled_frame_names = [all_frames[i] for i in indices]
        
        frames = []
        for name in sampled_frame_names:
            img_path = os.path.join(video_folder, name)
            img = Image.open(img_path).convert("RGB")
            frames.append(img)
            
        return frames

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # 1. Load exactly 50 sampled PIL images
        pil_images = self._load_and_sample_frames(video_id)
        
        if pil_images == None:
            return None
        
        # 2. Process images through CLIP processor to get the pixel values tensor
        # Shape output will be: (50, 3, 224, 224)
        pixel_values = self.clip_processor(images=pil_images, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # 3. Load the corresponding pre-calculated Wav2CLIP embedding for this video
        # (Assuming you saved them as torch tensors or numpy arrays named 'video_id.pt')
        audio_emb_path = os.path.join(self.audio_dir, f"{video_id}.npy")

        try:
            # Load the numpy array
            audio_np = np.load(audio_emb_path) 
            # Convert directly to a torch FloatTensor
            audio_embedding = torch.from_numpy(audio_np).float() 
        except Exception as e:
            print(f"Error loading audio embedding for {video_id}: {e}")
            return None
        
        return {
            "video_id": video_id,
            "pixel_values": pixel_values, 
            "audio_embedding": audio_embedding
        }
    

def safe_collate_fn(batch):
    # Filter out the None entries completely
    batch = [item for item in batch if item is not None]
    
    # If a batch accidentally ends up completely empty, handle gracefully
    if len(batch) == 0:
        return {}
        
    return {
        "video_id": [item["video_id"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "audio_embedding": torch.stack([item["audio_embedding"] for item in batch])
    }


class VideoAudioAttentionBridge(nn.Module):



    def __init__(self, clip_dim=512, audio_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        
        # 1. Temporal Attention Mechanism
        # This processes your (batch, 50, 512) tensor across the 50 frames
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=clip_dim, 
            nhead=num_heads, 
            dim_feedforward=clip_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. Geometric Projection Head 
        # This bends your collapsed visual sequence space directly into the Wav2CLIP neighborhood
        self.projection_head = nn.Sequential(
            nn.Linear(clip_dim, clip_dim * 2),
            nn.LayerNorm(clip_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(clip_dim * 2, audio_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Visual frame embeddings of shape (batch_size, 50, 512)
        Returns:
            projected_embedding (torch.Tensor): Perfectly aligned video vector of shape (batch_size, 512)
        """
        # Pass through the temporal transformer layers to calculate cross-frame dynamics
        attended_frames = self.temporal_transformer(x) # Shape remains: (batch, 50, 512)
        
        # Pool across the sequence length (50 frames) to derive a single visual vector
        # Mean pooling *after* attention captures the holistic event sequence contextually
        collapsed_video_vector = attended_frames.mean(dim=1) # Shape: (batch, 512)
        
        # Project vector space directly into the matching audio modality target coordinates
        projected_embedding = self.projection_head(collapsed_video_vector) # Shape: (batch, 512)
        
        # Normalize vectors to unit length so Cosine Similarity is a simple dot product
        return F.normalize(projected_embedding, p=2, dim=-1)