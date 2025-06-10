# Datasets Module Documentation

This document provides a comprehensive overview of the datasets module located in `src/datasets`. The module focuses on managing, processing, and transforming various human pose and motion datasets for training and evaluating pose estimation models.

## Table of Contents

1. [Overview](#overview)
2. [Current Module Components](#current-module-components)
3. [Planned Extensions](#planned-extensions)
4. [File Descriptions](#file-descriptions)
5. [Key Classes and Functions](#key-classes-and-functions)
6. [Dataset Descriptions](#dataset-descriptions)
7. [Usage Examples](#usage-examples)

## Overview

The datasets module provides functionality for loading, processing, and transforming various human pose and motion datasets. The primary goal is to standardize data from different sources into a common format that can be used for training pose estimation models and for constructing domain-specific knowledge models. The module integrates with Ray for distributed processing to handle large datasets efficiently.

## Current Module Components

The module currently consists of several components:

- **COCO Dataset Handling**: Classes for loading and processing the COCO keypoints dataset
- **Data Transformations**: Classes for preprocessing and augmenting pose data
- **Benchmark Generation**: Utilities for generating benchmark data for pose analysis
- **Motion Capture Parsing**: Parsers for AMC and C3D motion capture file formats
- **Human Pose Datasets**: Parsers for Max Planck human pose and NOMO 3D body scan datasets

## Planned Extensions

The module will be extended to handle the following features:

- **Custom Dataset Integration**: Creating a unified API for all datasets
- **Ollama Integration**: Building a continual learning pipeline with Ollama as the base model
- **Visualizations**: Enhanced visualization tools for different pose formats

## File Descriptions

### `coco.py`

Contains classes for loading and processing the COCO keypoints dataset, which is a common benchmark for human pose estimation. Includes classes for both training and validation datasets, as well as utilities for generating keypoint heatmaps and Part Affinity Fields (PAFs).

### `transformations.py`

Implements various data transformations for augmenting pose data, including scaling, rotation, cropping, and flipping. These transformations help improve model generalization by increasing the diversity of training data.

### `generate_benchmark.py`

Contains utilities for generating benchmark data for pose analysis using AI APIs (OpenAI and Anthropic). This helps create a gold standard dataset for evaluating the quality of pose analysis.

### `amc_parser.py`

Implements parsing and processing for Acclaim Motion Capture (AMC) files from the Carnegie University Digital Library. AMC files contain motion capture data that can be used for training and validating pose estimation models. The parser handles skeleton definitions and motion data, supporting parallel processing using Ray.

### `c3d_parser.py`

Implements parsing and processing for C3D (Coordinate 3D) files from the Carnegie University Digital Library. C3D is a binary file format used for recording 3D motion capture data. The parser extracts header information, point data, and analog data, providing a standardized interface to access motion capture information.

### `maxplanck_dataset.py`

Implements loading and processing for the Max Planck Human Pose Dataset, which contains images of people performing various activities with annotated 2D body joint positions. The implementation handles dataset extraction, annotation loading, and conversion to a standardized format compatible with other datasets in the module.

### `nomo_dataset.py`

Implements loading and processing for the NOMO Dataset, which contains 3D body scans and measurements. The parser supports multiple 3D scan formats (OBJ, PLY, STL), extracts anthropometric measurements, and can generate standardized skeleton keypoints from 3D body scans. It includes visualization capabilities and distributed processing using Ray.

### Planned New Files

#### `unified_dataset.py`

Will implement a unified API for all datasets, allowing for consistent data loading, preprocessing, and augmentation across different sources.

#### `ollama_trainer.py`

Will implement a continual learning pipeline using Ollama as the base model, enabling continuous improvement of the pose analysis model with new data.

## Key Classes and Functions

### `CocoTrainDataset` (in `coco.py`)

A PyTorch Dataset class for loading and processing COCO keypoints training data.

```python
class CocoTrainDataset(torch.utils.data.Dataset):
    def __init__(self, prepared_train_labels, images_folder, stride, sigma, paf_thickness, transform=None):
        """
        Initialize the COCO training dataset.
        
        Args:
            prepared_train_labels: Preprocessed annotations
            images_folder: Path to images
            stride: Model stride (output size = input size / stride)
            sigma: Gaussian sigma for keypoint heatmaps
            paf_thickness: Thickness for part affinity fields
            transform: Optional transformations to apply
        """
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns a dictionary containing:
            - image: The input image
            - keypoint_maps: Target heatmaps for keypoints
            - keypoint_mask: Mask for keypoint heatmaps
            - paf_maps: Target part affinity fields
            - paf_mask: Mask for part affinity fields
        """
```

### `DistributedCocoTrainDataset` (in `coco.py`)

A Ray-distributed version of the COCO training dataset for parallel processing.

```python
@ray.remote
class DistributedCocoTrainDataset(Dataset):
    def __init__(self, labels, images_folder, stride, sigma, paf_thickness, transform=None):
        """
        Initialize the distributed COCO training dataset.
        
        Args:
            labels: Path to the labels file
            images_folder: Path to images
            stride: Model stride (output size = input size / stride)
            sigma: Gaussian sigma for keypoint heatmaps
            paf_thickness: Thickness for part affinity fields
            transform: Optional transformations to apply
        """
        
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        
    def __len__(self):
        """Return the number of samples in the dataset."""
```

### Transformation Classes (in `transformations.py`)

A collection of classes for data augmentation.

```python
@ray.remote
class Scale:
    """Scale image and keypoints"""
    def __init__(self, scale_min=0.8, scale_max=1.2):
        """Initialize with scale range."""
        
    def __call__(self, sample):
        """Apply scaling to the sample."""

@ray.remote
class Rotate:
    """Rotate image and keypoints"""
    def __init__(self, max_rotation=30, pad=(0, 0, 0)):
        """Initialize with max rotation angle and padding value."""
        
    def __call__(self, sample):
        """Apply rotation to the sample."""

@ray.remote
class CropPad:
    """Crop and pad image to maintain size while centering content"""
    def __init__(self, pad=(0, 0, 0), target_size=None):
        """Initialize with padding value and target size."""
        
    def __call__(self, sample):
        """Apply cropping and padding to the sample."""

@ray.remote
class Flip:
    """Horizontally flip image and keypoints with 50% probability"""
    def __call__(self, sample):
        """Apply horizontal flipping to the sample."""
```

### `BenchmarkGenerator` (in `generate_benchmark.py`)

A class for generating benchmark data for pose analysis.

```python
class BenchmarkGenerator:
    def __init__(self):
        """Initialize the benchmark generator with API clients."""
        
    def generate_visualization_queries(self, pose_data, sport_type=None):
        """Generate visualization-specific queries for the pose data."""
        
    def generate_benchmark_data(self, model_path, sport_type=None, output_dir="benchmark_data"):
        """Generate benchmark data for a pose model using AI APIs."""
        
    def process_all_models(self, model_dir="models", sport_type=None, output_dir="benchmark_data"):
        """Process all model files in a directory to generate benchmark data."""
```

### `AMCParser` (in `amc_parser.py`)

A class for parsing and processing AMC motion capture files.

```python
class AMCParser:
    def __init__(self, data_folder):
        """
        Initialize the AMC parser.
        
        Args:
            data_folder: Path to the folder containing AMC files and skeleton definitions
        """
        
    def load_skeleton(self, asf_file):
        """
        Load and parse a skeleton definition from an ASF file.
        
        Args:
            asf_file: Path to the ASF file
            
        Returns:
            Dictionary containing the skeleton definition
        """
        
    def parse_motion(self, amc_file, skeleton=None):
        """
        Parse a motion file in AMC format.
        
        Args:
            amc_file: Path to the AMC file
            skeleton: Optional skeleton definition to use (if not provided, will try to find one)
            
        Returns:
            Dictionary containing the parsed motion data
        """
        
    def get_standard_keypoints(self, motion_data, frame_idx=0):
        """
        Convert motion data to standard keypoints format.
        
        Args:
            motion_data: Parsed motion data
            frame_idx: Index of the frame to extract
            
        Returns:
            Array of standardized keypoints
        """
        
    def process_all_motions(self, output_dir=None, num_cpus=None):
        """
        Process all motion files in the data folder using Ray for distributed computing.
        
        Args:
            output_dir: Directory to save processed data (optional)
            num_cpus: Number of CPUs to use for parallel processing
            
        Returns:
            List of processed motion data
        """
```

### `C3DParser` (in `c3d_parser.py`)

A class for parsing and processing C3D motion capture files.

```python
class C3DParser:
    def __init__(self, data_folder):
        """
        Initialize the C3D parser.
        
        Args:
            data_folder: Path to the folder containing C3D files
        """
        
    def list_motions(self):
        """
        List all available motion files.
        
        Returns:
            List of motion file paths
        """
        
    def parse_c3d(self, c3d_file):
        """
        Parse a C3D file.
        
        Args:
            c3d_file: Path to the C3D file
            
        Returns:
            Dictionary containing the parsed data
        """
        
    def get_standard_keypoints(self, c3d_data, frame_idx=0):
        """
        Convert C3D data to standard keypoints format.
        
        Args:
            c3d_data: Parsed C3D data
            frame_idx: Index of the frame to extract
            
        Returns:
            Array of standardized keypoints
        """
        
    def process_all_motions(self, output_dir=None, num_cpus=None):
        """
        Process all C3D files in the data folder using Ray for distributed computing.
        
        Args:
            output_dir: Directory to save processed data (optional)
            num_cpus: Number of CPUs to use for parallel processing
            
        Returns:
            List of processed motion data
        """
```

### `MaxPlanckDataset` (in `maxplanck_dataset.py`)

A class for loading and processing the Max Planck Human Pose Dataset.

```python
class MaxPlanckDataset:
    def __init__(self, dataset_folder):
        """
        Initialize the Max Planck dataset parser.
        
        Args:
            dataset_folder: Path to the folder containing the dataset
        """
        
    def _check_and_extract_dataset(self):
        """Extract the dataset from archives if needed."""
        
    def load_annotations(self):
        """
        Load and parse the annotations.
        
        Returns:
            Dictionary containing parsed annotations
        """
        
    def get_training_samples(self):
        """
        Get the training samples.
        
        Returns:
            List of training sample indices
        """
        
    def get_validation_samples(self):
        """
        Get the validation samples.
        
        Returns:
            List of validation sample indices
        """
        
    def get_sample_image(self, index):
        """
        Load an image for a specific sample.
        
        Args:
            index: Index of the sample
            
        Returns:
            The loaded image
        """
        
    def visualize_sample(self, index, output_path=None):
        """
        Visualize a sample with keypoints and bounding box.
        
        Args:
            index: Index of the sample
            output_path: Path to save the visualization (optional)
        """
        
    def convert_to_standard_format(self, output_file):
        """
        Convert the dataset to a standardized format.
        
        Args:
            output_file: Path to save the standardized dataset
        """
```

### `NOMODataset` (in `nomo_dataset.py`)

A class for loading and processing the NOMO 3D body scan dataset.

```python
class NOMODataset:
    def __init__(self, dataset_folder):
        """
        Initialize the NOMO dataset parser.
        
        Args:
            dataset_folder: Path to the folder containing the dataset
        """
        
    def _check_and_extract_dataset(self):
        """Extract the dataset from archives if needed."""
        
    def load_measurements(self):
        """
        Load and parse the measurement data.
        
        Returns:
            Dictionary containing parsed measurement data
        """
        
    def load_scan(self, subject_id):
        """
        Load a 3D scan for a specific subject.
        
        Args:
            subject_id: ID of the subject
            
        Returns:
            Dictionary containing scan data
        """
        
    def get_subject_info(self, subject_id):
        """
        Get all available information for a specific subject.
        
        Args:
            subject_id: ID of the subject
            
        Returns:
            Dictionary containing subject information
        """
        
    def visualize_scan(self, subject_id, output_path=None):
        """
        Visualize a 3D scan for a specific subject.
        
        Args:
            subject_id: ID of the subject
            output_path: Path to save the visualization (optional)
        """
        
    def generate_standard_keypoints(self, subject_id):
        """
        Generate standardized keypoints from a 3D scan.
        
        Args:
            subject_id: ID of the subject
            
        Returns:
            Dictionary containing standardized keypoint data
        """
        
    def process_all_subjects(self, output_dir=None):
        """
        Process all subjects in the dataset using Ray for distributed computing.
        
        Args:
            output_dir: Directory to save processed data (optional)
            
        Returns:
            Dictionary mapping subject IDs to processed data
        """
```

## Dataset Descriptions

### Carnegie University Digital Library (CUD)

Located in `public/datasources/CUD`, this dataset contains motion capture data in two formats:

- **AMC**: Acclaim Motion Capture format, which stores motion capture data as a series of joint angles and includes skeleton definition files (ASF) and motion files (AMC)
- **C3D**: Coordinate 3D format, which stores motion capture data as 3D coordinates with marker information

The dataset includes running and sprinting motions, which can be used for training and validation of human pose estimation models. The AMC format provides detailed skeletal structure with joint hierarchies, while C3D provides raw marker positions captured during motion performances.

### Max Planck Human Pose Dataset

Located in `public/datasources/MAXPLANCK`, this dataset contains images of people performing various activities with annotated 2D body joint positions. It includes annotations for 14 body joints, with visibility flags, bounding boxes, and activity labels. The dataset is divided into training and validation sets, making it suitable for supervised learning of pose estimation models.

### NOMO Dataset

Located in `public/datasources/NOMO`, this dataset contains 3D body scans and anthropometric measurements for a diverse population. The dataset provides:

- 3D body scans in various formats (OBJ, PLY, STL)
- Detailed measurements for body parts and proportions
- Demographic information (age, gender, height, weight)

This dataset is valuable for understanding human body proportions, which can help improve pose estimation models by providing anatomical constraints and priors.

## Usage Examples

### Loading and Processing COCO Dataset

```python
from src.datasets.coco import CocoTrainDataset
from src.datasets.transformations import Scale, Rotate, CropPad, Flip

# Create a transformation pipeline
transform = [
    Scale(scale_min=0.8, scale_max=1.2),
    Rotate(max_rotation=30),
    CropPad(target_size=(256, 192)),
    Flip()
]

# Create the dataset
dataset = CocoTrainDataset(
    prepared_train_labels="path/to/annotations.json",
    images_folder="path/to/images",
    stride=8,
    sigma=7,
    paf_thickness=1,
    transform=transform
)

# Access a sample
sample = dataset[0]
image = sample['image']
keypoint_maps = sample['keypoint_maps']
```

### Generating Benchmark Data

```python
from src.datasets.generate_benchmark import BenchmarkGenerator

# Create the benchmark generator
generator = BenchmarkGenerator()

# Generate benchmark data for a single model
benchmark_data = generator.generate_benchmark_data(
    model_path="path/to/model.pth",
    sport_type="running",
    output_dir="benchmark_data"
)

# Process all models in a directory
all_benchmark_data = generator.process_all_models(
    model_dir="models",
    sport_type="running",
    output_dir="benchmark_data"
)
```

### Parsing AMC Motion Capture Files

```python
from src.datasets.amc_parser import AMCParser

# Create the AMC parser
parser = AMCParser(data_folder="public/datasources/CUD/amc")

# List available motions
motions = parser.list_motions()
print(f"Found {len(motions)} motion files")

# Parse a single motion
motion_data = parser.parse_motion(motions[0])
print(f"Parsed motion with {len(motion_data['frames'])} frames")

# Extract standard keypoints for the first frame
keypoints = parser.get_standard_keypoints(motion_data, frame_idx=0)
print(f"Extracted {len(keypoints)} keypoints")

# Process all motions in parallel
processed_data = parser.process_all_motions(
    output_dir="processed/amc",
    num_cpus=4
)
print(f"Processed {len(processed_data)} motion files")
```

### Parsing C3D Motion Capture Files

```python
from src.datasets.c3d_parser import C3DParser

# Create the C3D parser
parser = C3DParser(data_folder="public/datasources/CUD/c3d")

# List available motions
motions = parser.list_motions()
print(f"Found {len(motions)} motion files")

# Parse a single C3D file
c3d_data = parser.parse_c3d(motions[0])
print(f"Parsed C3D file with {c3d_data['point']['frames']} frames and {c3d_data['point']['count']} markers")

# Extract standard keypoints for the first frame
keypoints = parser.get_standard_keypoints(c3d_data, frame_idx=0)
print(f"Extracted {len(keypoints)} keypoints")

# Process all C3D files in parallel
processed_data = parser.process_all_motions(
    output_dir="processed/c3d",
    num_cpus=4
)
print(f"Processed {len(processed_data)} C3D files")
```

### Working with the Max Planck Dataset

```python
from src.datasets.maxplanck_dataset import MaxPlanckDataset

# Create the dataset parser
dataset = MaxPlanckDataset(dataset_folder="public/datasources/MAXPLANCK")

# Load annotations
annotations = dataset.load_annotations()
print(f"Loaded annotations for {len(annotations['image_paths'])} images")

# Get training and validation samples
train_samples = dataset.get_training_samples()
val_samples = dataset.get_validation_samples()
print(f"Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

# Visualize a sample
dataset.visualize_sample(train_samples[0], output_path="output/maxplanck_sample.png")

# Convert to standard format
dataset.convert_to_standard_format(output_file="processed/maxplanck_standard.json")
```

### Working with the NOMO Dataset

```python
from src.datasets.nomo_dataset import NOMODataset

# Create the dataset parser
dataset = NOMODataset(dataset_folder="public/datasources/NOMO")

# Load measurements
measurements = dataset.load_measurements()
print(f"Loaded measurements for {len(measurements['subject_id'])} subjects")

# Get info for a specific subject
subject_id = measurements['subject_id'][0]
subject_info = dataset.get_subject_info(subject_id)
print(f"Subject {subject_id} - Gender: {subject_info['gender']}, Height: {subject_info['height']} cm")

# Visualize a subject's 3D scan
dataset.visualize_scan(subject_id, output_path="output/nomo_sample.png")

# Generate standard keypoints from a 3D scan
keypoints = dataset.generate_standard_keypoints(subject_id)
print(f"Generated {keypoints['keypoints'].shape[0]} keypoints for subject {subject_id}")

# Process all subjects
processed_data = dataset.process_all_subjects(output_dir="processed/NOMO")
print(f"Processed {len(processed_data)} subjects")
```
