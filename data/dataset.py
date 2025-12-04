"""
Dataset filtering module for Something-Something V2.
Filters annotations to three core tasks: move_object, drop_object, cover_object.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import re
import random


# Task definitions based on template patterns
MOVE_OBJECT_TEMPLATES = [
    "Moving something up",
    "Moving something down",
    "Moving something across a surface without it falling down",
    "Moving something towards the camera",
    "Moving something away from the camera",
    "Pulling something from right to left",
    "Pulling something from left to right",
    "Pushing something from right to left",
    "Pushing something from left to right",
]

DROP_OBJECT_TEMPLATES = [
    "Dropping something onto something",
    "Dropping something in front of something",
    "Dropping something behind something",
    "Dropping something into something",
    "Dropping something next to something",
     "Something falling like a rock",
     "Something falling like a feather or paper",
]

COVER_OBJECT_TEMPLATES = [
    "Covering something with something",
    "Putting something on a surface",
    "Putting something onto something",
    "Putting something on a flat surface without letting it roll",
    "Putting something on the edge of something so it is not supported and falls down"
]

BACK_AND_FORTH_TEMPLATES = [
    "Letting something roll up a slanted surface, so it rolls back down",
    "Plugging something into something but pulling it right out as you remove your hand",
    "Throwing something in the air and catching it",
]




def load_annotations(annotations_path: Path) -> List[Dict]:
    """Load annotations from JSON file."""
    with open(annotations_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def match_template(template: str, target_templates: List[str]) -> bool:
    """
    Check if a template matches any of the target templates.
    Data templates have [something] placeholders, our target templates don't.
    Simply check if target is in the template after normalizing placeholders.
    """
    # Normalize: replace [something] with 'something' and lowercase
    normalized = template.replace('[something]', 'something').lower()
    
    for target in target_templates:
        # Check if the target pattern appears in the normalized template
        if target.lower() in normalized:
            return True
    
    return False


def filter_by_task(annotations: List[Dict], task: str) -> List[Dict]:
    """
    Filter annotations by task type.
    
    Args:
        annotations: List of annotation dictionaries
        task: One of 'move_object', 'drop_object', 'cover_object'
    
    Returns:
        Filtered list of annotations
    """
    if task == 'move_object':
        templates = MOVE_OBJECT_TEMPLATES
    elif task == 'drop_object':
        templates = DROP_OBJECT_TEMPLATES
    elif task == 'cover_object':
        templates = COVER_OBJECT_TEMPLATES
    elif task == 'back_and_forth':
        templates = BACK_AND_FORTH_TEMPLATES
    else:
        raise ValueError(f"Unknown task: {task}. Must be one of: move_object, drop_object, cover_object")
    
    filtered = []
    for ann in annotations:
        template = ann.get('template', '')
        if match_template(template, templates):
            filtered.append(ann)
    
    return filtered


def filter_dataset(
    train_json_path: Path,
    val_json_path: Path,
    output_dir: Path,
    tasks: List[str] = ['move_object', 'drop_object', 'cover_object'],
    train_samples_per_task: int = 1000,
    val_samples_per_task: int = 100
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter dataset to specified tasks, sample specified number per task, and save filtered annotations.
    
    Args:
        train_json_path: Path to train.json
        val_json_path: Path to validation.json
        output_dir: Directory to save filtered annotations
        tasks: List of tasks to include
        train_samples_per_task: Number of samples to sample per task from training set
        val_samples_per_task: Number of samples to sample per task from validation set
    
    Returns:
        Tuple of (filtered_train, filtered_val) annotations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {train_json_path}...")
    train_annotations = load_annotations(train_json_path)
    print(f"Loaded {len(train_annotations)} training annotations")
    
    print(f"Loading annotations from {val_json_path}...")
    val_annotations = load_annotations(val_json_path)
    print(f"Loaded {len(val_annotations)} validation annotations")
    
    # Filter and sample by tasks
    filtered_train = []
    filtered_val = []
    
    for task in tasks:
        print(f"\nFiltering {task}...")
        train_task = filter_by_task(train_annotations, task)
        val_task = filter_by_task(val_annotations, task)
        
        print(f"  Train: {len(train_task)} samples (before sampling)")
        print(f"  Val: {len(val_task)} samples (before sampling)")
        
        # Sample from training set
        if len(train_task) > train_samples_per_task:
            random.shuffle(train_task)
            train_task = train_task[:train_samples_per_task]
            print(f"  Sampled {train_samples_per_task} training samples")
        else:
            print(f"  Using all {len(train_task)} training samples (less than {train_samples_per_task})")
        
        # Sample from validation set
        if len(val_task) > val_samples_per_task:
            random.shuffle(val_task)
            val_task = val_task[:val_samples_per_task]
            print(f"  Sampled {val_samples_per_task} validation samples")
        else:
            print(f"  Using all {len(val_task)} validation samples (less than {val_samples_per_task})")
        
        # Add task label to each annotation
        for ann in train_task:
            ann['task'] = task
        for ann in val_task:
            ann['task'] = task
        
        filtered_train.extend(train_task)
        filtered_val.extend(val_task)
    
    print(f"\nTotal filtered samples:")
    print(f"  Train: {len(filtered_train)}")
    print(f"  Val: {len(filtered_val)}")
    
    # Save filtered annotations
    if tasks[0] == "back_and_forth":
        filtered_train_path = output_dir / 'train_filtered_backforth.json'
        filtered_val_path = output_dir / 'val_filtered_backforth.json'
    else:
        filtered_train_path = output_dir / 'train_filtered.json'
        filtered_val_path = output_dir / 'val_filtered.json'
        
    with open(filtered_train_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_train, f, indent=2, ensure_ascii=False)
    print(f"\nSaved filtered training annotations to {filtered_train_path}")
    
    with open(filtered_val_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_val, f, indent=2, ensure_ascii=False)
    print(f"Saved filtered validation annotations to {filtered_val_path}")
    
    return filtered_train, filtered_val


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    base_dir = Path(__file__).parent / "sthv2"
    train_json = base_dir / "annotations" / "train.json"
    val_json = base_dir / "annotations" / "validation.json"
    output_dir = base_dir / "annotations"
    
    # filter basic movements (move drop cover)
    filter_dataset(
        train_json, 
        val_json, 
        output_dir,
        train_samples_per_task=1000,
        val_samples_per_task=100
    )

    # filter back and forth movements
    filter_dataset(
        train_json,
        val_json,
        output_dir,
        tasks=['back_and_forth'],
        train_samples_per_task=3000,
        val_samples_per_task=300
    )

