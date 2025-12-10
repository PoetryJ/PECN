"""
Video frame extraction module.
Extracts all frames from videos in two resolutions: 96x96 and 128x128.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_frame(video_path: Path, frame_index: int, target_size: Tuple[int, int] = (96, 96)) -> Optional[np.ndarray]:
    """
    Extract a specific frame from a video.
    
    Args:
        video_path: Path to video file
        frame_index: 0-based frame index (frame 20 = index 20)
        target_size: Target size (width, height) for resizing
    
    Returns:
        Frame as numpy array (RGB, uint8) or None if frame doesn't exist
    """
    if not video_path.exists():
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_index >= total_frames:
        cap.release()
        return None
    
    # Seek to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    frame_resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    return frame_resized


def extract_all_frames(
    video_path: Path,
    target_size: Tuple[int, int] = (96, 96)
) -> List[Optional[np.ndarray]]:
    """
    Extract all frames from a video.
    
    Args:
        video_path: Path to video file
        target_size: Target size (width, height) for resizing
    
    Returns:
        List of frames as numpy arrays (RGB, uint8), or empty list if extraction fails
    """
    if not video_path.exists():
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        frame_resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        
        frames.append(frame_resized)
    
    cap.release()
    return frames


def save_frame(frame: np.ndarray, output_path: Path) -> None:
    """Save a frame as an image file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame)
    img.save(output_path)


def _process_single_video(
    ann: dict,
    videos_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int],
    cache_frames: bool
) -> Optional[dict]:
    """
    Process a single video: extract and cache all frames.
    
    Returns:
        Updated annotation dict with frame info, or None if processing failed
    """
    video_id = ann['id']
    video_path = videos_dir / f"{video_id}.webm"
    
    if not video_path.exists():
        return None
    
    # Extract all frames
    frames = extract_all_frames(video_path, target_size)
    
    if len(frames) == 0:
        return None
    
    # Cache frames if requested
    if cache_frames:
        for frame_idx, frame in enumerate(frames):
            frame_path = output_dir / f"{video_id}_frame_{frame_idx:05d}.png"
            save_frame(frame, frame_path)
    
    # Update annotation with frame info
    ann['num_frames'] = len(frames)
    ann['video_path'] = str(video_path)
    
    return ann


def extract_frames_for_annotations(
    annotations: list,
    videos_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (96, 96),
    cache_frames: bool = True,
    num_workers: int = 8
) -> list:
    """
    Extract all frames from videos for all annotations and cache them (multi-threaded).
    
    Args:
        annotations: List of annotation dictionaries with 'id' field
        videos_dir: Directory containing video files
        output_dir: Directory to save cached frames (if cache_frames=True)
        target_size: Target size for resizing (width, height)
        cache_frames: Whether to cache extracted frames to disk
        num_workers: Number of parallel threads (default: 8)
    
    Returns:
        List of valid annotations (those with successfully extracted frames)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_annotations = []
    
    print(f"Extracting all frames for {len(annotations)} videos...")
    print(f"Target size: {target_size}, Workers: {num_workers}")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_ann = {
            executor.submit(
                _process_single_video,
                ann.copy(),  # Copy to avoid race conditions
                videos_dir,
                output_dir,
                target_size,
                cache_frames
            ): ann for ann in annotations
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_ann), total=len(annotations), desc="Extracting frames"):
            result = future.result()
            if result is not None:
                valid_annotations.append(result)
    
    print(f"\nSuccessfully extracted frames for {len(valid_annotations)}/{len(annotations)} videos")
    
    return valid_annotations


def load_cached_frame(frame_path: Path) -> Optional[np.ndarray]:
    """Load a cached frame from disk."""
    if not frame_path.exists():
        return None
    
    img = Image.open(frame_path)
    return np.array(img)


if __name__ == "__main__":
    import sys
    import os
    
    # Import from same package
    try:
        from .dataset import load_annotations
    except ImportError:
        from dataset import load_annotations
    
    base_dir = Path(__file__).parent.parent / "data" / "sthv2"
    filtered_train_path = base_dir / "annotations" / "train_filtered.json"
    filtered_val_path = base_dir / "annotations" / "val_filtered.json"
    filtered_train_backforth_path = base_dir / "annotations" / "train_filtered_backforth.json"
    filtered_val_backforth_path = base_dir / "annotations" / "val_filtered_backforth.json"
    videos_dir = base_dir / "videos"
    
    # Load train and val annotations
    train_annotations = load_annotations(filtered_train_path)
    val_annotations = load_annotations(filtered_val_path)
    train_backforth_annotations = load_annotations(filtered_train_backforth_path)
    val_backforth_annotations = load_annotations(filtered_val_backforth_path)
    print(f"Loaded {len(train_annotations)} training annotations")
    print(f"Loaded {len(val_annotations)} validation annotations")
    print(f"Loaded {len(train_backforth_annotations)} training backforth annotations")
    print(f"Loaded {len(val_backforth_annotations)} validation backforth annotations")
    
    num_workers = int(os.environ.get('NUM_WORKERS', '8'))  # Default: 8 threads
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Parallel workers: {num_workers}")
    print(f"{'='*60}")
    
    # Extract frames for both resolutions
    resolutions = [(224, 224)]
    
    # We'll update annotations with num_frames after first resolution
    updated_train_annotations = None
    updated_val_annotations = None
    
    for idx, (width, height) in enumerate(resolutions):
        print(f"\n{'='*60}")
        print(f"Extracting frames at {width}x{height} resolution...")
        print(f"{'='*60}")
        
        output_dir = base_dir / f"frames_{width}x{height}"
        
        # Extract training frames
        if train_annotations:
            print(f"\nProcessing training set ({len(train_annotations)} videos)...")
            
            valid_train = extract_frames_for_annotations(
                train_annotations,
                videos_dir,
                output_dir,
                target_size=(width, height),
                cache_frames=True,
                num_workers=num_workers
            )
            print(f"Successfully processed {len(valid_train)}/{len(train_annotations)} training videos")
            
            # Save updated annotations with num_frames after first resolution
            if idx == 0:
                updated_train_annotations = valid_train
        
        # Extract validation frames
        if val_annotations:
            print(f"\nProcessing validation set ({len(val_annotations)} videos)...")
            
            valid_val = extract_frames_for_annotations(
                val_annotations,
                videos_dir,
                output_dir,
                target_size=(width, height),
                cache_frames=True,
                num_workers=num_workers
            )
            print(f"Successfully processed {len(valid_val)}/{len(val_annotations)} validation videos")
            
            # Save updated annotations with num_frames after first resolution
            if idx == 0:
                updated_val_annotations = valid_val

        # Extract training backforth frames
        if train_backforth_annotations:
            print(f"\nProcessing training backforth set ({len(train_backforth_annotations)} videos)...")
            
            valid_train_backforth = extract_frames_for_annotations(
                train_backforth_annotations,
                videos_dir,
                output_dir,
                target_size=(width, height),
                cache_frames=True,
                num_workers=num_workers
            )
            print(f"Successfully processed {len(valid_train_backforth)}/{len(train_backforth_annotations)} training backforth videos")
            
            # Save updated annotations with num_frames after first resolution
            if idx == 0:
                updated_train_backforth_annotations = valid_train_backforth

        # Extract validation backforth frames
        if val_backforth_annotations:
            print(f"\nProcessing validation backforth set ({len(val_backforth_annotations)} videos)...")
            
            valid_val_backforth = extract_frames_for_annotations(
                val_backforth_annotations,
                videos_dir,
                output_dir,
                target_size=(width, height),
                cache_frames=True,
                num_workers=num_workers
            )
            print(f"Successfully processed {len(valid_val_backforth)}/{len(val_backforth_annotations)} validation backforth videos")
            
            # Save updated annotations with num_frames after first resolution
            if idx == 0:
                updated_val_backforth_annotations = valid_val_backforth
    
    # Save updated annotations with num_frames back to JSON files
    if updated_train_annotations:
        print(f"\nSaving updated training annotations with num_frames to {filtered_train_path}...")
        with open(filtered_train_path, 'w', encoding='utf-8') as f:
            json.dump(updated_train_annotations, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(updated_train_annotations)} training annotations")
    
    if updated_val_annotations:
        print(f"Saving updated validation annotations with num_frames to {filtered_val_path}...")
        with open(filtered_val_path, 'w', encoding='utf-8') as f:
            json.dump(updated_val_annotations, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(updated_val_annotations)} validation annotations")

    if updated_train_backforth_annotations:
        print(f"Saving updated training backforth annotations with num_frames to {filtered_train_backforth_path}...")
        with open(filtered_train_backforth_path, 'w', encoding='utf-8') as f:
            json.dump(updated_train_backforth_annotations, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(updated_train_backforth_annotations)} training backforth annotations")

    if updated_val_backforth_annotations:
        print(f"Saving updated validation backforth annotations with num_frames to {filtered_val_backforth_path}...")
        with open(filtered_val_backforth_path, 'w', encoding='utf-8') as f:
            json.dump(updated_val_backforth_annotations, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(updated_val_backforth_annotations)} validation backforth annotations")

    print(f"\n{'='*60}")
    print("Frame extraction completed!")
    print(f"{'='*60}")

