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


def create_sthv2_dataset(
    annotations_path: Path,
    frames_dir: Path,
    input_frame_idx: int = 20,
    target_frame_idx: int = 21,
    split: str = "train"
) -> Dataset:
    """
    Create a HuggingFace Dataset from STHV2 annotations for ControlNet training.
    
    Args:
        annotations_path: Path to filtered annotations JSON
        frames_dir: Directory containing cached frames (e.g., frames_96x96 or frames_128x128)
        input_frame_idx: Frame index for conditioning image (frame 20)
        target_frame_idx: Frame index for target image (frame 21)
        split: Dataset split name
    
    Returns:
        HuggingFace Dataset compatible with official training scripts
    """
    # Load annotations
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Prepare data
    data = {
        'image': [],  # target image (frame 21)
        'conditioning_image': [],  # input image (frame 20)
        'text': [],  # text description
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
            data['image'].append(str(target_frame_path))
            data['conditioning_image'].append(str(input_frame_path))
            data['text'].append(ann.get('label', ann.get('template', '')))
            valid_samples.append(ann)
    
    print(f"Created {split} dataset with {len(valid_samples)} samples")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(data)
    
    # Cast image columns to Image type
    dataset = dataset.cast_column('image', HFImage())
    dataset = dataset.cast_column('conditioning_image', HFImage())
    
    return dataset


def create_sthv2_instruct_pix2pix_dataset(
    annotations_path: Path,
    frames_dir: Path,
    input_frame_idx: int = 20,
    target_frame_idx: int = 21,
    split: str = "train"
) -> Dataset:
    """
    Create a HuggingFace Dataset from STHV2 annotations for InstructPix2Pix training.
    
    Args:
        annotations_path: Path to filtered annotations JSON
        frames_dir: Directory containing cached frames (e.g., frames_96x96 or frames_128x128)
        input_frame_idx: Frame index for original image (frame 20)
        target_frame_idx: Frame index for edited image (frame 21)
        split: Dataset split name
    
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
            # Format as instruction: "Predict the next frame after: {action}"
            text = ann.get('label', ann.get('template', ''))
            data['edit_prompt'].append(f"Show the next frame after: {text}")
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
    
    # Create ControlNet dataset (using 96x96 frames)
    print("Creating ControlNet dataset (96x96)...")
    train_dataset = create_sthv2_dataset(
        annotations_path=base_dir / "annotations" / "train_filtered.json",
        frames_dir=base_dir / "frames_96x96",
        split="train"
    )
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Features: {train_dataset.features}")
    if len(train_dataset) > 0:
        print(f"Example: {train_dataset[0]}")
    
    # Create InstructPix2Pix dataset (using 96x96 frames)
    print("\nCreating InstructPix2Pix dataset (96x96)...")
    train_dataset_ip2p = create_sthv2_instruct_pix2pix_dataset(
        annotations_path=base_dir / "annotations" / "train_filtered.json",
        frames_dir=base_dir / "frames_96x96",
        split="train"
    )
    print(f"Train dataset: {len(train_dataset_ip2p)} samples")
    print(f"Features: {train_dataset_ip2p.features}")

