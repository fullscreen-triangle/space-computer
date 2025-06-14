#!/usr/bin/env python3
"""
Test script for the pose understanding verification system

This script demonstrates how the pose understanding verification works
and provides examples of successful and failed verification scenarios.
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from backend.core.pose_understanding import (
    PoseUnderstandingVerifier,
    verify_pose_understanding_before_analysis,
    save_verification_images
)

# Sample pose data for testing
SAMPLE_POSES = {
    "standing_neutral": {
        "nose": {"x": 0.5, "y": 0.3, "confidence": 0.9},
        "left_shoulder": {"x": 0.4, "y": 0.5, "confidence": 0.8},
        "right_shoulder": {"x": 0.6, "y": 0.5, "confidence": 0.8},
        "left_elbow": {"x": 0.35, "y": 0.7, "confidence": 0.7},
        "right_elbow": {"x": 0.65, "y": 0.7, "confidence": 0.7},
        "left_wrist": {"x": 0.3, "y": 0.9, "confidence": 0.6},
        "right_wrist": {"x": 0.7, "y": 0.9, "confidence": 0.6},
        "left_hip": {"x": 0.45, "y": 0.8, "confidence": 0.8},
        "right_hip": {"x": 0.55, "y": 0.8, "confidence": 0.8},
        "left_knee": {"x": 0.44, "y": 1.1, "confidence": 0.7},
        "right_knee": {"x": 0.56, "y": 1.1, "confidence": 0.7},
        "left_ankle": {"x": 0.43, "y": 1.4, "confidence": 0.6},
        "right_ankle": {"x": 0.57, "y": 1.4, "confidence": 0.6}
    },
    
    "arms_raised": {
        "nose": {"x": 0.5, "y": 0.3, "confidence": 0.9},
        "left_shoulder": {"x": 0.4, "y": 0.5, "confidence": 0.8},
        "right_shoulder": {"x": 0.6, "y": 0.5, "confidence": 0.8},
        "left_elbow": {"x": 0.3, "y": 0.3, "confidence": 0.7},
        "right_elbow": {"x": 0.7, "y": 0.3, "confidence": 0.7},
        "left_wrist": {"x": 0.2, "y": 0.1, "confidence": 0.6},
        "right_wrist": {"x": 0.8, "y": 0.1, "confidence": 0.6},
        "left_hip": {"x": 0.45, "y": 0.8, "confidence": 0.8},
        "right_hip": {"x": 0.55, "y": 0.8, "confidence": 0.8},
        "left_knee": {"x": 0.44, "y": 1.1, "confidence": 0.7},
        "right_knee": {"x": 0.56, "y": 1.1, "confidence": 0.7},
        "left_ankle": {"x": 0.43, "y": 1.4, "confidence": 0.6},
        "right_ankle": {"x": 0.57, "y": 1.4, "confidence": 0.6}
    },
    
    "crouching": {
        "nose": {"x": 0.5, "y": 0.5, "confidence": 0.9},
        "left_shoulder": {"x": 0.4, "y": 0.7, "confidence": 0.8},
        "right_shoulder": {"x": 0.6, "y": 0.7, "confidence": 0.8},
        "left_elbow": {"x": 0.35, "y": 0.9, "confidence": 0.7},
        "right_elbow": {"x": 0.65, "y": 0.9, "confidence": 0.7},
        "left_wrist": {"x": 0.3, "y": 1.1, "confidence": 0.6},
        "right_wrist": {"x": 0.7, "y": 1.1, "confidence": 0.6},
        "left_hip": {"x": 0.45, "y": 1.0, "confidence": 0.8},
        "right_hip": {"x": 0.55, "y": 1.0, "confidence": 0.8},
        "left_knee": {"x": 0.4, "y": 0.8, "confidence": 0.7},
        "right_knee": {"x": 0.6, "y": 0.8, "confidence": 0.7},
        "left_ankle": {"x": 0.43, "y": 1.2, "confidence": 0.6},
        "right_ankle": {"x": 0.57, "y": 1.2, "confidence": 0.6}
    },
    
    "corrupted_data": {
        "nose": {"x": 0.5, "y": 0.3, "confidence": 0.1},
        "left_shoulder": {"x": -1.0, "y": 2.0, "confidence": 0.2},
        "right_shoulder": {"x": 3.0, "y": -0.5, "confidence": 0.1},
        # Missing other joints to simulate poor data
    }
}

SAMPLE_QUERIES = [
    "What is the angle of the left elbow?",
    "How is the posture of this person?",
    "Analyze the biomechanics of this pose",
    "Compare this to optimal technique",
    "What muscles are most active in this position?"
]

async def test_single_verification(verifier, pose_name, pose_data, query):
    """Test verification for a single pose"""
    print(f"\n{'='*60}")
    print(f"Testing: {pose_name}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = await verifier.verify_understanding(pose_data, query)
        
        print(f"✅ Verification completed in {result.verification_time:.2f}s")
        print(f"   Understanding: {'✅ YES' if result.understood else '❌ NO'}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Similarity Score: {result.similarity_score:.3f}")
        
        if result.error_message:
            print(f"   Error: {result.error_message}")
        
        # Save verification images for debugging
        if result.generated_image is not None:
            output_dir = f"debug/verification_test/{pose_name}"
            save_verification_images(result, output_dir, f"test_{pose_name}")
            print(f"   Debug images saved to: {output_dir}")
        
        return result
        
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        return None

async def test_batch_verification(verifier, poses, queries):
    """Test batch verification with multiple poses and queries"""
    print(f"\n{'='*60}")
    print("BATCH VERIFICATION TEST")
    print(f"{'='*60}")
    
    results = []
    
    for pose_name, pose_data in poses.items():
        for query in queries[:2]:  # Test with first 2 queries
            result = await test_single_verification(verifier, pose_name, pose_data, query)
            if result:
                results.append({
                    'pose': pose_name,
                    'query': query,
                    'understood': result.understood,
                    'confidence': result.confidence,
                    'similarity': result.similarity_score,
                    'time': result.verification_time
                })
    
    return results

def analyze_results(results):
    """Analyze and summarize test results"""
    if not results:
        print("\n❌ No results to analyze")
        return
    
    print(f"\n{'='*60}")
    print("RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    total_tests = len(results)
    successful = sum(1 for r in results if r['understood'])
    success_rate = successful / total_tests * 100
    
    avg_confidence = sum(r['confidence'] for r in results) / total_tests
    avg_similarity = sum(r['similarity'] for r in results) / total_tests
    avg_time = sum(r['time'] for r in results) / total_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful Verifications: {successful}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Average Similarity: {avg_similarity:.3f}")
    print(f"Average Verification Time: {avg_time:.2f}s")
    
    # Breakdown by pose type
    print(f"\n📊 Results by Pose Type:")
    pose_stats = {}
    for result in results:
        pose = result['pose']
        if pose not in pose_stats:
            pose_stats[pose] = {'total': 0, 'successful': 0, 'confidence': []}
        
        pose_stats[pose]['total'] += 1
        if result['understood']:
            pose_stats[pose]['successful'] += 1
        pose_stats[pose]['confidence'].append(result['confidence'])
    
    for pose, stats in pose_stats.items():
        success_rate = stats['successful'] / stats['total'] * 100
        avg_conf = sum(stats['confidence']) / len(stats['confidence'])
        print(f"   {pose}: {success_rate:.1f}% success, {avg_conf:.3f} avg confidence")

async def test_convenience_function():
    """Test the convenience function for integration"""
    print(f"\n{'='*60}")
    print("TESTING CONVENIENCE FUNCTION")
    print(f"{'='*60}")
    
    pose_data = SAMPLE_POSES["standing_neutral"]
    query = "Analyze this standing pose"
    
    should_proceed, result = await verify_pose_understanding_before_analysis(
        pose_data, query
    )
    
    print(f"Should proceed with analysis: {'✅ YES' if should_proceed else '❌ NO'}")
    if result:
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Similarity: {result.similarity_score:.3f}")

async def main():
    """Main test function"""
    print("🚀 Starting Pose Understanding Verification Tests")
    print("=" * 60)
    
    # Initialize verifier
    print("Initializing pose understanding verifier...")
    verifier = PoseUnderstandingVerifier(similarity_threshold=0.7)
    
    try:
        await verifier.initialize()
        print("✅ Verifier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize verifier: {e}")
        print("This might be due to missing dependencies (diffusers, clip, etc.)")
        print("The system will work but verification will be disabled.")
        return
    
    # Test individual verifications
    results = await test_batch_verification(verifier, SAMPLE_POSES, SAMPLE_QUERIES)
    
    # Analyze results
    analyze_results(results)
    
    # Test convenience function
    await test_convenience_function()
    
    # Test configuration changes
    print(f"\n{'='*60}")
    print("TESTING CONFIGURATION CHANGES")
    print(f"{'='*60}")
    
    # Test with different threshold
    verifier.similarity_threshold = 0.5
    print("Testing with lower threshold (0.5)...")
    result = await verifier.verify_understanding(
        SAMPLE_POSES["standing_neutral"], 
        "Test with lower threshold"
    )
    print(f"Result with lower threshold: {'✅' if result.understood else '❌'}")
    
    print(f"\n🎉 All tests completed!")
    print("Check the debug/verification_test/ directory for generated images")

if __name__ == "__main__":
    # Create debug directory
    os.makedirs("debug/verification_test", exist_ok=True)
    
    # Run tests
    asyncio.run(main()) 