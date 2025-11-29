"""
HuggingFace Datasets-compatible wrapper for STHV2 dataset.
Compatible with official diffusers training scripts.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value



def create_sthv2_instruct_pix2pix_dataset(
    annotations_path: Path,
    frames_dir: Path,
    input_frame_idx: int = 20,
    target_frame_idx: int = 21,
    split: str = "train",
    add_progress: bool = False
) -> Dataset:
    """
    Create a HuggingFace Dataset from STHV2 annotations for InstructPix2Pix training.
    
    Args:
        annotations_path: Path to filtered annotations JSON
        frames_dir: Directory containing cached frames (e.g., frames_96x96 or frames_128x128)
        input_frame_idx: Frame index for original image (frame 20)
        target_frame_idx: Frame index for edited image (frame 21)
        split: Dataset split name
        add_progress: Whether to add progress percentage to edit prompt
    
    Returns:
        HuggingFace Dataset compatible with InstructPix2Pix training
    """
    # Load annotations
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Prepare data
    data = {
        'input_image': [],  # original image (frame 20)
        'edited_image': [],  # target image (frame 21)
        'edit_prompt': [],  # text instruction
    }
    
    frames_dir = Path(frames_dir)
    valid_samples = []
    
    for ann in annotations:
        video_id = ann['id']
        # Frame files are named as {video_id}_frame_{frame_idx:05d}.png
        input_frame_path = frames_dir / f"{video_id}_frame_{input_frame_idx:05d}.png"
        target_frame_path = frames_dir / f"{video_id}_frame_{target_frame_idx:05d}.png"
        
        # Check if both frames exist
        if input_frame_path.exists() and target_frame_path.exists():
            data['input_image'].append(str(input_frame_path))
            data['edited_image'].append(str(target_frame_path))
            # Format as instruction: "Generate a future frame of this action: {action}"
            text = ann.get('label', ann.get('template', ''))
            prompt = f"Generate a future frame of this action: {text}"
            
            # Add progress information if requested
            if add_progress and 'num_frames' in ann:
                num_frames = ann['num_frames']
                if num_frames > 0:
                    progress = int((input_frame_idx / num_frames) * 10) * 10
                    prompt += f" And {progress}% of the action has been completed."
            
            data['edit_prompt'].append(prompt)
            valid_samples.append(ann)
    
    print(f"Created {split} dataset with {len(valid_samples)} samples")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(data)
    
    # Cast image columns to Image type
    dataset = dataset.cast_column('input_image', HFImage())
    dataset = dataset.cast_column('edited_image', HFImage())
    
    return dataset


if __name__ == "__main__":
    # Example usage
    base_dir = Path(__file__).parent.parent / "data" / "sthv2"
    
    # Create InstructPix2Pix dataset (using 96x96 frames)
    print("\nCreating InstructPix2Pix dataset (96x96)...")
    train_dataset_ip2p = create_sthv2_instruct_pix2pix_dataset(
        annotations_path=base_dir / "annotations" / "train_filtered.json",
        frames_dir=base_dir / "frames_96x96",
        split="train"
    )
    print(f"Train dataset: {len(train_dataset_ip2p)} samples")
    print(f"Features: {train_dataset_ip2p.features}")

