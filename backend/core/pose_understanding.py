"""
Pose Understanding Verification System

This module ensures AI comprehension of pose data by requiring the AI to generate
visual representations of poses before providing analysis. This prevents garbage
responses by validating understanding through image generation.
"""

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import logging
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline
import base64
import io
from sklearn.metrics.pairwise import cosine_similarity
import clip

logger = logging.getLogger(__name__)

@dataclass
class PoseUnderstandingResult:
    """Result of pose understanding verification"""
    understood: bool
    confidence: float
    generated_image: Optional[np.ndarray]
    similarity_score: float
    verification_time: float
    error_message: Optional[str] = None

class PoseSkeletonRenderer:
    """Renders pose skeletons for comparison"""
    
    # Standard pose connections (COCO format)
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    JOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        
    def render_skeleton(self, pose_data: Dict[str, Dict[str, float]], 
                       background_color: Tuple[int, int, int] = (255, 255, 255),
                       skeleton_color: Tuple[int, int, int] = (0, 0, 255),
                       joint_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """Render pose skeleton as image"""
        
        # Create blank image
        image = np.full((*self.image_size, 3), background_color, dtype=np.uint8)
        
        # Convert pose data to pixel coordinates
        keypoints = self._normalize_pose_to_image(pose_data)
        
        # Draw connections
        for connection in self.POSE_CONNECTIONS:
            if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]
                
                if pt1 is not None and pt2 is not None:
                    cv2.line(image, pt1, pt2, skeleton_color, 2)
        
        # Draw joints
        for point in keypoints:
            if point is not None:
                cv2.circle(image, point, 4, joint_color, -1)
                
        return image
    
    def _normalize_pose_to_image(self, pose_data: Dict[str, Dict[str, float]]) -> List[Optional[Tuple[int, int]]]:
        """Convert pose coordinates to image pixel coordinates"""
        keypoints = []
        
        # Find bounding box of pose
        valid_points = [(v['x'], v['y']) for v in pose_data.values() 
                       if 'x' in v and 'y' in v and v.get('confidence', 0) > 0.3]
        
        if not valid_points:
            return [None] * len(self.JOINT_NAMES)
        
        xs, ys = zip(*valid_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Add padding
        padding = 0.1
        width = max_x - min_x
        height = max_y - min_y
        min_x -= width * padding
        max_x += width * padding
        min_y -= height * padding
        max_y += height * padding
        
        # Scale to image size
        scale_x = (self.image_size[0] - 40) / (max_x - min_x) if max_x != min_x else 1
        scale_y = (self.image_size[1] - 40) / (max_y - min_y) if max_y != min_y else 1
        scale = min(scale_x, scale_y)
        
        # Convert each joint
        for joint_name in self.JOINT_NAMES:
            if joint_name in pose_data:
                joint = pose_data[joint_name]
                if joint.get('confidence', 0) > 0.3:
                    x = int((joint['x'] - min_x) * scale + 20)
                    y = int((joint['y'] - min_y) * scale + 20)
                    keypoints.append((x, y))
                else:
                    keypoints.append(None)
            else:
                keypoints.append(None)
                
        return keypoints

class AIImageGenerator:
    """Generates images from text descriptions using AI"""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        self.model_name = model_name
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Initialize the image generation pipeline"""
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()
            logger.info(f"Image generation pipeline initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize image generation pipeline: {e}")
            raise
    
    async def generate_pose_image(self, pose_description: str, 
                                 negative_prompt: str = "blurry, distorted, multiple people") -> Optional[np.ndarray]:
        """Generate image from pose description"""
        if not self.pipeline:
            await self.initialize()
            
        try:
            # Enhanced prompt for better pose generation
            enhanced_prompt = f"simple stick figure drawing, {pose_description}, clean white background, black lines, anatomically correct pose, single person"
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipeline(
                    enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=512,
                    height=512
                )
            
            # Convert to numpy array
            image = np.array(result.images[0])
            return image
            
        except Exception as e:
            logger.error(f"Failed to generate pose image: {e}")
            return None

class PoseDescriptionGenerator:
    """Generates text descriptions of poses for AI image generation"""
    
    def __init__(self):
        self.joint_descriptions = {
            'nose': 'head',
            'left_shoulder': 'left shoulder', 'right_shoulder': 'right shoulder',
            'left_elbow': 'left elbow', 'right_elbow': 'right elbow',
            'left_wrist': 'left hand', 'right_wrist': 'right hand',
            'left_hip': 'left hip', 'right_hip': 'right hip',
            'left_knee': 'left knee', 'right_knee': 'right knee',
            'left_ankle': 'left foot', 'right_ankle': 'right foot'
        }
    
    def generate_description(self, pose_data: Dict[str, Dict[str, float]]) -> str:
        """Generate natural language description of pose"""
        descriptions = []
        
        # Analyze arm positions
        arm_desc = self._describe_arms(pose_data)
        if arm_desc:
            descriptions.append(arm_desc)
            
        # Analyze leg positions
        leg_desc = self._describe_legs(pose_data)
        if leg_desc:
            descriptions.append(leg_desc)
            
        # Analyze overall posture
        posture_desc = self._describe_posture(pose_data)
        if posture_desc:
            descriptions.append(posture_desc)
        
        if not descriptions:
            return "person standing in neutral position"
            
        return "stick figure with " + ", ".join(descriptions)
    
    def _describe_arms(self, pose_data: Dict[str, Dict[str, float]]) -> str:
        """Describe arm positions"""
        descriptions = []
        
        # Check if arms are raised
        if self._is_joint_above(pose_data, 'left_wrist', 'left_shoulder'):
            descriptions.append("left arm raised")
        if self._is_joint_above(pose_data, 'right_wrist', 'right_shoulder'):
            descriptions.append("right arm raised")
            
        # Check if arms are extended
        if self._are_joints_far(pose_data, 'left_shoulder', 'left_wrist'):
            descriptions.append("left arm extended")
        if self._are_joints_far(pose_data, 'right_shoulder', 'right_wrist'):
            descriptions.append("right arm extended")
            
        return ", ".join(descriptions) if descriptions else ""
    
    def _describe_legs(self, pose_data: Dict[str, Dict[str, float]]) -> str:
        """Describe leg positions"""
        descriptions = []
        
        # Check if legs are bent
        if self._is_joint_bent(pose_data, 'left_hip', 'left_knee', 'left_ankle'):
            descriptions.append("left leg bent")
        if self._is_joint_bent(pose_data, 'right_hip', 'right_knee', 'right_ankle'):
            descriptions.append("right leg bent")
            
        # Check stance width
        if self._are_joints_far(pose_data, 'left_ankle', 'right_ankle'):
            descriptions.append("wide stance")
            
        return ", ".join(descriptions) if descriptions else ""
    
    def _describe_posture(self, pose_data: Dict[str, Dict[str, float]]) -> str:
        """Describe overall posture"""
        # Check if person is crouching
        if (self._is_joint_below(pose_data, 'left_hip', 'left_knee') or 
            self._is_joint_below(pose_data, 'right_hip', 'right_knee')):
            return "crouching posture"
            
        # Check if person is leaning
        if self._is_leaning(pose_data):
            return "leaning posture"
            
        return ""
    
    def _is_joint_above(self, pose_data: Dict, joint1: str, joint2: str) -> bool:
        """Check if joint1 is above joint2"""
        if joint1 not in pose_data or joint2 not in pose_data:
            return False
        return pose_data[joint1].get('y', 0) < pose_data[joint2].get('y', 0)
    
    def _is_joint_below(self, pose_data: Dict, joint1: str, joint2: str) -> bool:
        """Check if joint1 is below joint2"""
        if joint1 not in pose_data or joint2 not in pose_data:
            return False
        return pose_data[joint1].get('y', 0) > pose_data[joint2].get('y', 0)
    
    def _are_joints_far(self, pose_data: Dict, joint1: str, joint2: str, threshold: float = 0.3) -> bool:
        """Check if joints are far apart"""
        if joint1 not in pose_data or joint2 not in pose_data:
            return False
        
        j1, j2 = pose_data[joint1], pose_data[joint2]
        distance = np.sqrt((j1.get('x', 0) - j2.get('x', 0))**2 + 
                          (j1.get('y', 0) - j2.get('y', 0))**2)
        return distance > threshold
    
    def _is_joint_bent(self, pose_data: Dict, joint1: str, joint2: str, joint3: str) -> bool:
        """Check if joint2 forms a bent angle between joint1 and joint3"""
        if not all(j in pose_data for j in [joint1, joint2, joint3]):
            return False
            
        # Calculate angle at joint2
        j1, j2, j3 = pose_data[joint1], pose_data[joint2], pose_data[joint3]
        
        v1 = np.array([j1.get('x', 0) - j2.get('x', 0), j1.get('y', 0) - j2.get('y', 0)])
        v2 = np.array([j3.get('x', 0) - j2.get('x', 0), j3.get('y', 0) - j2.get('y', 0)])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        
        return angle < 150  # Consider bent if angle is less than 150 degrees
    
    def _is_leaning(self, pose_data: Dict) -> bool:
        """Check if person is leaning"""
        if 'nose' not in pose_data or 'left_hip' not in pose_data or 'right_hip' not in pose_data:
            return False
            
        # Calculate center of hips
        left_hip = pose_data['left_hip']
        right_hip = pose_data['right_hip']
        hip_center_x = (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2
        
        # Check if head is significantly offset from hip center
        head_x = pose_data['nose'].get('x', 0)
        offset = abs(head_x - hip_center_x)
        
        return offset > 0.1  # Threshold for leaning

class ImageSimilarityCalculator:
    """Calculates similarity between images using CLIP embeddings"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        
    async def initialize(self):
        """Initialize CLIP model for image similarity"""
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model initialized for image similarity")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise
    
    async def calculate_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate similarity between two images"""
        if self.model is None:
            await self.initialize()
            
        try:
            # Convert numpy arrays to PIL Images
            img1 = Image.fromarray(image1)
            img2 = Image.fromarray(image2)
            
            # Preprocess images
            img1_tensor = self.preprocess(img1).unsqueeze(0).to(self.device)
            img2_tensor = self.preprocess(img2).unsqueeze(0).to(self.device)
            
            # Get image embeddings
            with torch.no_grad():
                img1_features = self.model.encode_image(img1_tensor)
                img2_features = self.model.encode_image(img2_tensor)
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(img1_features, img2_features).item()
                
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Failed to calculate image similarity: {e}")
            return 0.0

class PoseUnderstandingVerifier:
    """Main class for verifying AI understanding of poses through image generation"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.skeleton_renderer = PoseSkeletonRenderer()
        self.image_generator = AIImageGenerator()
        self.description_generator = PoseDescriptionGenerator()
        self.similarity_calculator = ImageSimilarityCalculator()
        
    async def initialize(self):
        """Initialize all components"""
        await self.image_generator.initialize()
        await self.similarity_calculator.initialize()
        logger.info("Pose understanding verifier initialized")
    
    async def verify_understanding(self, pose_data: Dict[str, Dict[str, float]], 
                                 query: str) -> PoseUnderstandingResult:
        """
        Verify AI understanding of pose by generating image and comparing similarity
        
        Args:
            pose_data: Dictionary containing pose joint positions
            query: The user's query about the pose
            
        Returns:
            PoseUnderstandingResult with verification details
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Render the actual pose skeleton
            actual_skeleton = self.skeleton_renderer.render_skeleton(pose_data)
            
            # Step 2: Generate description of the pose
            pose_description = self.description_generator.generate_description(pose_data)
            logger.info(f"Generated pose description: {pose_description}")
            
            # Step 3: Have AI generate image from description
            generated_image = await self.image_generator.generate_pose_image(pose_description)
            
            if generated_image is None:
                return PoseUnderstandingResult(
                    understood=False,
                    confidence=0.0,
                    generated_image=None,
                    similarity_score=0.0,
                    verification_time=asyncio.get_event_loop().time() - start_time,
                    error_message="Failed to generate image"
                )
            
            # Step 4: Calculate similarity between actual and generated
            similarity_score = await self.similarity_calculator.calculate_similarity(
                actual_skeleton, generated_image
            )
            
            # Step 5: Determine if understanding is sufficient
            understood = similarity_score >= self.similarity_threshold
            confidence = min(1.0, similarity_score / self.similarity_threshold)
            
            verification_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Pose understanding verification: similarity={similarity_score:.3f}, "
                       f"understood={understood}, time={verification_time:.2f}s")
            
            return PoseUnderstandingResult(
                understood=understood,
                confidence=confidence,
                generated_image=generated_image,
                similarity_score=similarity_score,
                verification_time=verification_time
            )
            
        except Exception as e:
            logger.error(f"Error in pose understanding verification: {e}")
            return PoseUnderstandingResult(
                understood=False,
                confidence=0.0,
                generated_image=None,
                similarity_score=0.0,
                verification_time=asyncio.get_event_loop().time() - start_time,
                error_message=str(e)
            )
    
    async def verify_with_retry(self, pose_data: Dict[str, Dict[str, float]], 
                               query: str, max_retries: int = 2) -> PoseUnderstandingResult:
        """Verify understanding with retry logic"""
        
        for attempt in range(max_retries + 1):
            result = await self.verify_understanding(pose_data, query)
            
            if result.understood or attempt == max_retries:
                if attempt > 0:
                    logger.info(f"Pose understanding achieved after {attempt + 1} attempts")
                return result
            
            logger.warning(f"Pose understanding failed (attempt {attempt + 1}/{max_retries + 1}), "
                          f"similarity: {result.similarity_score:.3f}")
            
            # Brief delay before retry
            await asyncio.sleep(1.0)
        
        return result

# Utility functions for integration
async def verify_pose_understanding_before_analysis(pose_data: Dict[str, Dict[str, float]], 
                                                   query: str,
                                                   verifier: Optional[PoseUnderstandingVerifier] = None) -> Tuple[bool, PoseUnderstandingResult]:
    """
    Convenience function to verify pose understanding before proceeding with analysis
    
    Returns:
        Tuple of (should_proceed, verification_result)
    """
    if verifier is None:
        verifier = PoseUnderstandingVerifier()
        await verifier.initialize()
    
    result = await verifier.verify_with_retry(pose_data, query)
    
    return result.understood, result

def save_verification_images(result: PoseUnderstandingResult, output_dir: str, prefix: str = "verification"):
    """Save verification images for debugging"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if result.generated_image is not None:
        generated_path = os.path.join(output_dir, f"{prefix}_generated.png")
        cv2.imwrite(generated_path, cv2.cvtColor(result.generated_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved generated image to {generated_path}") 