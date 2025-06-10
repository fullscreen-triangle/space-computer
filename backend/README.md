# Spectacular Backend

This directory contains the backend implementation for the Spectacular project, including all AI models and processing pipelines.

## Implemented Hugging Face Models

The following Hugging Face models have been implemented according to the recommendations in `ai.md`:

### 2D Pose Estimation
- Primary: `ultralytics/yolov8s-pose` - Fast pose estimation with 17 keypoints
- Fallback: `qualcomm/RTMPose_Body2d` - Mobile-optimized model with 133 keypoints
- Implementation: `backend/core/pose_pipeline.py`

### 3D Pose & Motion Embedding
- `walterzhu/MotionBERT-Lite` - Lifts 2D poses to 3D and generates motion embeddings
- `Tonic/video-swin-transformer` - RGB-based motion features and phase segmentation
- Implementation: `backend/core/pose3d.py` and `backend/core/video_feat.py`

### Action Recognition
- Skeleton-based: Custom classifier head on MotionBERT embeddings
- RGB-based: Fine-tuned Video Swin Transformer for phase and technique classification
- Implementation: `backend/core/action_head.py` and `backend/core/action_rgb.py`

### Retrieval Embeddings (RAG)
- Primary: `sentence-transformers/all-MiniLM-L6-v2` - General-purpose document embeddings
- Optional: `allenai/scibert_scivocab_uncased` - Academic corpus similarity & citation lookup
- Implementation: `backend/llm/embeddings.py`

### Voice Processing
- ASR: `openai/whisper-large-v3` - Robust streaming speech-to-text
- TTS: `coqui/XTTS-v2` - Voice cloning with multilingual support
- Implementation: `backend/voice/asr.py` and `backend/voice/tts.py`

### Vision-Language Captioning
- `Salesforce/blip2-flan-t5-xl` - Auto-captioning for richer LLM context
- Implementation: `backend/vision/caption.py`

### LLM
- Default: Mistral-7B-Instruct
- Upgrade Option: `meta-llama/Meta-Llama-3-8B-Instruct` - Improved reasoning
- Implementation: `backend/llm/model.py`

## Usage

Each module is designed to be used independently or as part of a pipeline. All models follow a similar initialization pattern:

```python
# Example for pose estimation
from backend.core.pose_pipeline import PosePipeline

# Initialize the model (falls back to CPU if CUDA not available)
pose_model = PosePipeline(use_fallback=False)

# Use the model
results = pose_model.infer_2d(image)
```

## Requirements

All necessary dependencies are listed in the project's `requirements.txt` file. Make sure to install them before using the modules.

## Notes

- Some models require access to Hugging Face's gated models (particularly Llama-3).
- Models are automatically quantized to 4-bit precision when possible for efficient inference.
- For production use, consider fine-tuning the action recognition models on your specific data. 