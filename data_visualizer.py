#!/usr/bin/env python3
"""
Biomechanical Data Visualizer
============================
Interactive visualization tool for exploring the datasources folder content
before Space Computer integration.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib.animation import FuncAnimation
import cv2
from datetime import datetime

class BiomechanicalDataVisualizer:
    def __init__(self, datasources_path="datasources"):
        self.datasources_path = Path(datasources_path)
        self.athletes = {}
        self.current_athlete = None
        
        # Set up visualization style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        print(f"🏃‍♂️ Biomechanical Data Visualizer")
        print(f"📁 Data source: {self.datasources_path.absolute()}")
        print("=" * 50)
        
    def scan_data_files(self):
        """Scan and catalog all available data files"""
        print("📊 Scanning data files...")
        
        # Scan pose data models
        models_path = self.datasources_path / "models"
        posture_path = self.datasources_path / "posture"
        annotated_path = self.datasources_path / "annotated"
        
        for file_path in models_path.glob("*_pose_data.json"):
            athlete_name = file_path.stem.replace("_pose_data", "")
            
            self.athletes[athlete_name] = {
                "name": athlete_name.replace("-", " ").replace("_", " ").title(),
                "pose_data_file": file_path,
                "biomechanics_file": None,
                "motion_file": None,
                "video_file": None,
                "sport": self._detect_sport(athlete_name)
            }
            
            # Check for corresponding files
            video_file = annotated_path / f"{athlete_name}_annotated.mp4"
            if video_file.exists():
                self.athletes[athlete_name]["video_file"] = video_file
                
        # Scan biomechanics files
        for file_path in posture_path.glob("*_biomechanics_analysis.json"):
            base_name = file_path.stem.replace("_biomechanics_analysis", "")
            if base_name in self.athletes:
                self.athletes[base_name]["biomechanics_file"] = file_path
                
        # Scan motion analysis files
        for file_path in posture_path.glob("*_motion_analysis.json"):
            base_name = file_path.stem.replace("_motion_analysis", "")
            if base_name in self.athletes:
                self.athletes[base_name]["motion_file"] = file_path
        
        print(f"✅ Found {len(self.athletes)} athletes with data")
        return self.athletes
    
    def _detect_sport(self, athlete_name):
        """Detect sport based on athlete name"""
        sport_keywords = {
            "bolt": "Sprinting",
            "powell": "Sprinting", 
            "beijing": "Athletics",
            "drogba": "Football",
            "boundary": "Cricket",
            "chisora": "Boxing",
            "pound": "Boxing",
            "koroibete": "Rugby",
            "lomu": "Rugby",
            "struggle": "Wrestling",
            "hezvo": "Track & Field"
        }
        
        for keyword, sport in sport_keywords.items():
            if keyword in athlete_name.lower():
                return sport
        return "General"
    
    def list_athletes(self):
        """Display available athletes"""
        print("\n🏆 Available Athletes:")
        print("-" * 40)
        
        for i, (key, athlete) in enumerate(self.athletes.items(), 1):
            video_status = "✅" if athlete["video_file"] else "❌"
            bio_status = "✅" if athlete["biomechanics_file"] else "❌"
            
            print(f"{i:2d}. {athlete['name']:<20} | {athlete['sport']:<12} | Video: {video_status} | Bio: {bio_status}")
    
    def load_athlete_data(self, athlete_key):
        """Load all data for a specific athlete"""
        if athlete_key not in self.athletes:
            raise ValueError(f"Athlete {athlete_key} not found")
            
        athlete = self.athletes[athlete_key]
        self.current_athlete = athlete_key
        
        print(f"\n📊 Loading data for {athlete['name']}...")
        
        # Load pose data
        if athlete["pose_data_file"]:
            with open(athlete["pose_data_file"], 'r') as f:
                athlete["pose_data"] = json.load(f)
                print(f"   ✅ Pose data: {len(athlete['pose_data']['pose_data'])} frames")
        
        # Load biomechanics data
        if athlete["biomechanics_file"]:
            with open(athlete["biomechanics_file"], 'r') as f:
                athlete["biomechanics_data"] = json.load(f)
                frame_count = len(athlete["biomechanics_data"]["frames"])
                print(f"   ✅ Biomechanics: {frame_count} frames")
        
        # Load motion analysis data
        if athlete["motion_file"]:
            with open(athlete["motion_file"], 'r') as f:
                athlete["motion_data"] = json.load(f)
                print(f"   ✅ Motion analysis loaded")
                
        return athlete
    
    def analyze_pose_data(self, athlete_key=None):
        """Analyze and visualize pose data"""
        if athlete_key:
            athlete = self.load_athlete_data(athlete_key)
        else:
            athlete = self.athletes[self.current_athlete]
            
        if "pose_data" not in athlete:
            print("❌ No pose data available")
            return
            
        pose_data = athlete["pose_data"]
        video_info = pose_data["video_info"]
        frames_data = pose_data["pose_data"]
        
        print(f"\n📹 Video Analysis - {athlete['name']}")
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        print(f"   FPS: {video_info['fps']}")
        print(f"   Total Frames: {video_info['total_frames']}")
        print(f"   Duration: {video_info['total_frames']/video_info['fps']:.1f} seconds")
        
        # Analyze pose detection quality
        pose_counts = []
        confidence_scores = []
        
        for frame in frames_data:
            pose_counts.append(len(frame["poses"]))
            
            if frame["poses"]:
                # Calculate average confidence for visible landmarks
                landmarks = frame["poses"][0]["landmarks"]
                confidences = [lm["visibility"] for lm in landmarks if lm["visibility"] > 0.5]
                if confidences:
                    confidence_scores.append(np.mean(confidences))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Pose Analysis - {athlete['name']}", fontsize=16, fontweight='bold')
        
        # Plot 1: Pose detection over time
        axes[0,0].plot(range(len(pose_counts)), pose_counts, 'cyan', linewidth=2)
        axes[0,0].set_title("Pose Detection Count Over Time")
        axes[0,0].set_xlabel("Frame")
        axes[0,0].set_ylabel("Number of Poses Detected")
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Confidence scores
        if confidence_scores:
            axes[0,1].plot(confidence_scores, 'lime', linewidth=2)
            axes[0,1].set_title("Average Pose Confidence")
            axes[0,1].set_xlabel("Frame")
            axes[0,1].set_ylabel("Confidence Score")
            axes[0,1].set_ylim(0, 1)
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Pose quality histogram
        axes[1,0].hist(confidence_scores, bins=20, color='orange', alpha=0.7, edgecolor='white')
        axes[1,0].set_title("Pose Quality Distribution")
        axes[1,0].set_xlabel("Confidence Score")
        axes[1,0].set_ylabel("Frequency")
        
        # Plot 4: Sample pose visualization
        self._plot_sample_pose(axes[1,1], frames_data, athlete['name'])
        
        plt.tight_layout()
        plt.show()
        
        return {
            "total_frames": len(frames_data),
            "avg_poses_per_frame": np.mean(pose_counts),
            "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0,
            "pose_detection_rate": sum(1 for x in pose_counts if x > 0) / len(pose_counts)
        }
    
    def _plot_sample_pose(self, ax, frames_data, athlete_name):
        """Plot a sample pose from the middle of the sequence"""
        # Find a frame with good pose data
        for frame in frames_data[len(frames_data)//2:]:
            if frame["poses"]:
                landmarks = frame["poses"][0]["landmarks"]
                
                # Extract x, y coordinates
                x_coords = [lm["x"] for lm in landmarks if lm["visibility"] > 0.5]
                y_coords = [1 - lm["y"] for lm in landmarks if lm["visibility"] > 0.5]  # Flip Y
                
                if len(x_coords) > 10:  # Ensure we have enough points
                    ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.8)
                    
                    # Draw simple skeleton connections
                    skeleton_connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # Head chain
                        (5, 6),  # Shoulders
                        (5, 7), (7, 9),  # Left arm
                        (6, 8), (8, 10),  # Right arm
                        (11, 12),  # Hips
                        (11, 13), (13, 15),  # Left leg
                        (12, 14), (14, 16),  # Right leg
                        (5, 11), (6, 12)  # Torso
                    ]
                    
                    for start_idx, end_idx in skeleton_connections:
                        if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                            landmarks[start_idx]["visibility"] > 0.5 and
                            landmarks[end_idx]["visibility"] > 0.5):
                            
                            x1, y1 = landmarks[start_idx]["x"], 1 - landmarks[start_idx]["y"]
                            x2, y2 = landmarks[end_idx]["x"], 1 - landmarks[end_idx]["y"]
                            ax.plot([x1, x2], [y1, y2], 'cyan', linewidth=2, alpha=0.7)
                    
                    ax.set_title(f"Sample Pose - {athlete_name}")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_aspect('equal')
                    break
    
    def analyze_biomechanics(self, athlete_key=None):
        """Analyze biomechanical data"""
        if athlete_key:
            athlete = self.load_athlete_data(athlete_key)
        else:
            athlete = self.athletes[self.current_athlete]
            
        if "biomechanics_data" not in athlete:
            print("❌ No biomechanics data available")
            return
            
        bio_data = athlete["biomechanics_data"]["frames"]
        
        print(f"\n🦴 Biomechanical Analysis - {athlete['name']}")
        print(f"   Total frames: {len(bio_data)}")
        
        # Extract time series data
        frames = list(range(len(bio_data)))
        joint_angles = {
            'hip': [frame["joint_angles"]["hip"] for frame in bio_data],
            'knee': [frame["joint_angles"]["knee"] for frame in bio_data],
            'ankle': [frame["joint_angles"]["ankle"] for frame in bio_data],
            'shoulder': [frame["joint_angles"]["shoulder"] for frame in bio_data],
            'elbow': [frame["joint_angles"]["elbow"] for frame in bio_data]
        }
        
        ground_forces = {
            'vertical': [frame["ground_reaction"]["vertical"] for frame in bio_data],
            'horizontal': [frame["ground_reaction"]["horizontal"] for frame in bio_data],
            'impact': [frame["ground_reaction"]["impact_force"] for frame in bio_data]
        }
        
        stability_scores = [frame["balance_metrics"]["stability_score"] for frame in bio_data]
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Biomechanical Analysis - {athlete['name']}", fontsize=16, fontweight='bold')
        
        # Joint Angles Over Time
        for joint, angles in joint_angles.items():
            axes[0,0].plot(frames, angles, label=joint, linewidth=2)
        axes[0,0].set_title("Joint Angles Over Time")
        axes[0,0].set_xlabel("Frame")
        axes[0,0].set_ylabel("Angle (degrees)")
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Ground Reaction Forces
        axes[0,1].plot(frames, ground_forces['vertical'], 'lime', label='Vertical', linewidth=2)
        axes[0,1].plot(frames, ground_forces['horizontal'], 'orange', label='Horizontal', linewidth=2)
        axes[0,1].set_title("Ground Reaction Forces")
        axes[0,1].set_xlabel("Frame")
        axes[0,1].set_ylabel("Force (N)")
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Stability Score
        axes[0,2].plot(frames, stability_scores, 'cyan', linewidth=2)
        axes[0,2].fill_between(frames, stability_scores, alpha=0.3)
        axes[0,2].set_title("Balance & Stability")
        axes[0,2].set_xlabel("Frame")
        axes[0,2].set_ylabel("Stability Score")
        axes[0,2].set_ylim(0, 1)
        axes[0,2].grid(True, alpha=0.3)
        
        # Joint Angle Distributions
        joint_data = list(joint_angles.values())
        joint_names = list(joint_angles.keys())
        axes[1,0].boxplot(joint_data, labels=joint_names)
        axes[1,0].set_title("Joint Angle Distributions")
        axes[1,0].set_ylabel("Angle (degrees)")
        plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
        
        # Force Distribution
        axes[1,1].hist([ground_forces['vertical'], ground_forces['horizontal']], 
                       bins=20, alpha=0.7, label=['Vertical', 'Horizontal'], 
                       color=['lime', 'orange'])
        axes[1,1].set_title("Force Distribution")
        axes[1,1].set_xlabel("Force (N)")
        axes[1,1].set_ylabel("Frequency")
        axes[1,1].legend()
        
        # Performance Summary
        self._plot_performance_summary(axes[1,2], joint_angles, ground_forces, stability_scores, athlete['name'])
        
        plt.tight_layout()
        plt.show()
        
        # Statistical summary
        stats = {
            "joint_ranges": {joint: max(angles) - min(angles) for joint, angles in joint_angles.items()},
            "avg_stability": np.mean(stability_scores),
            "max_vertical_force": max(ground_forces['vertical']),
            "max_impact_force": max(ground_forces['impact']),
            "avg_horizontal_force": np.mean(np.abs(ground_forces['horizontal']))
        }
        
        print("\n📈 Performance Summary:")
        for metric, value in stats.items():
            if isinstance(value, dict):
                print(f"   {metric}:")
                for sub_metric, sub_value in value.items():
                    print(f"      {sub_metric}: {sub_value:.1f}°")
            else:
                print(f"   {metric}: {value:.2f}")
        
        return stats
    
    def _plot_performance_summary(self, ax, joint_angles, ground_forces, stability_scores, athlete_name):
        """Plot performance summary radar chart"""
        # Calculate normalized metrics (0-1 scale)
        metrics = {
            'Knee ROM': (max(joint_angles['knee']) - min(joint_angles['knee'])) / 180,
            'Hip ROM': (max(joint_angles['hip']) - min(joint_angles['hip'])) / 180,
            'Stability': np.mean(stability_scores),
            'Power': min(max(ground_forces['vertical']) / 2000, 1),  # Normalize to reasonable max
            'Balance': 1 - (np.std(stability_scores) / np.mean(stability_scores))  # Lower variability = better
        }
        
        # Simple bar chart instead of radar for easier implementation
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=['red', 'orange', 'yellow', 'lime', 'cyan'], alpha=0.7)
        ax.set_title(f"Performance Profile\n{athlete_name}")
        ax.set_ylabel("Normalized Score")
        ax.set_ylim(0, 1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    def compare_athletes(self, athlete_keys):
        """Compare multiple athletes"""
        if len(athlete_keys) < 2:
            print("❌ Need at least 2 athletes for comparison")
            return
            
        comparison_data = {}
        
        for key in athlete_keys:
            athlete = self.load_athlete_data(key)
            if "biomechanics_data" in athlete:
                bio_data = athlete["biomechanics_data"]["frames"]
                
                comparison_data[athlete['name']] = {
                    'avg_stability': np.mean([f["balance_metrics"]["stability_score"] for f in bio_data]),
                    'max_force': max([f["ground_reaction"]["impact_force"] for f in bio_data]),
                    'knee_rom': max([f["joint_angles"]["knee"] for f in bio_data]) - min([f["joint_angles"]["knee"] for f in bio_data]),
                    'sport': athlete['sport']
                }
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Athlete Comparison", fontsize=16, fontweight='bold')
        
        metrics = ['avg_stability', 'max_force', 'knee_rom']
        titles = ['Average Stability', 'Maximum Force (N)', 'Knee Range of Motion (°)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            names = list(comparison_data.keys())
            values = [comparison_data[name][metric] for name in names]
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            
            bars = axes[i].bar(names, values, color=colors, alpha=0.8)
            axes[i].set_title(title)
            axes[i].set_ylabel(title.split()[0])
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                           f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_data
    
    def generate_data_report(self):
        """Generate comprehensive data report"""
        print("\n" + "="*60)
        print("📊 BIOMECHANICAL DATA REPORT")
        print("="*60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Source: {self.datasources_path.absolute()}")
        
        print(f"\n📈 DATASET SUMMARY:")
        print(f"   Total Athletes: {len(self.athletes)}")
        
        # Count by sport
        sports = {}
        total_frames = 0
        
        for athlete_data in self.athletes.values():
            sport = athlete_data['sport']
            sports[sport] = sports.get(sport, 0) + 1
            
            # Count frames if data is loaded
            if "pose_data" in athlete_data:
                total_frames += len(athlete_data["pose_data"]["pose_data"])
        
        print(f"   Total Frames Analyzed: {total_frames:,}")
        print(f"   Sports Represented: {len(sports)}")
        
        for sport, count in sorted(sports.items()):
            print(f"      {sport}: {count} athletes")
        
        # Data completeness
        pose_count = sum(1 for a in self.athletes.values() if a.get("pose_data_file"))
        bio_count = sum(1 for a in self.athletes.values() if a.get("biomechanics_file"))
        video_count = sum(1 for a in self.athletes.values() if a.get("video_file"))
        
        print(f"\n📁 DATA COMPLETENESS:")
        print(f"   Pose Data: {pose_count}/{len(self.athletes)} ({pose_count/len(self.athletes)*100:.0f}%)")
        print(f"   Biomechanics: {bio_count}/{len(self.athletes)} ({bio_count/len(self.athletes)*100:.0f}%)")
        print(f"   Videos: {video_count}/{len(self.athletes)} ({video_count/len(self.athletes)*100:.0f}%)")
        
        print(f"\n🚀 INTEGRATION RECOMMENDATIONS:")
        print(f"   1. Start with athletes having complete data sets")
        print(f"   2. Focus on {max(sports, key=sports.get)} athletes for initial testing")
        print(f"   3. Use high-quality pose data for model validation")
        print(f"   4. Implement sport-specific analysis modules")
        
        return {
            "total_athletes": len(self.athletes),
            "total_frames": total_frames,
            "sports_distribution": sports,
            "data_completeness": {
                "pose": pose_count / len(self.athletes),
                "biomechanics": bio_count / len(self.athletes),
                "video": video_count / len(self.athletes)
            }
        }

def main():
    """Main execution function"""
    visualizer = BiomechanicalDataVisualizer()
    
    # Scan available data
    athletes = visualizer.scan_data_files()
    
    # Show menu
    while True:
        print("\n" + "="*50)
        print("🏃‍♂️ BIOMECHANICAL DATA EXPLORER")
        print("="*50)
        print("1. List all athletes")
        print("2. Analyze individual athlete")
        print("3. Compare athletes")
        print("4. Generate data report")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            visualizer.list_athletes()
            
        elif choice == "2":
            visualizer.list_athletes()
            athlete_num = input("\nSelect athlete number: ").strip()
            
            try:
                athlete_num = int(athlete_num) - 1
                athlete_keys = list(athletes.keys())
                
                if 0 <= athlete_num < len(athlete_keys):
                    selected_key = athlete_keys[athlete_num]
                    print(f"\n📊 Analyzing {athletes[selected_key]['name']}...")
                    
                    # Show analysis menu
                    print("\nAnalysis Options:")
                    print("1. Pose Analysis")
                    print("2. Biomechanical Analysis") 
                    print("3. Both")
                    
                    analysis_choice = input("Select analysis (1-3): ").strip()
                    
                    if analysis_choice in ["1", "3"]:
                        visualizer.analyze_pose_data(selected_key)
                    
                    if analysis_choice in ["2", "3"]:
                        visualizer.analyze_biomechanics(selected_key)
                        
                else:
                    print("❌ Invalid athlete number")
                    
            except ValueError:
                print("❌ Please enter a valid number")
                
        elif choice == "3":
            visualizer.list_athletes()
            athlete_nums = input("\nSelect athlete numbers (comma-separated, e.g., 1,2,3): ").strip()
            
            try:
                nums = [int(x.strip()) - 1 for x in athlete_nums.split(",")]
                athlete_keys = list(athletes.keys())
                selected_keys = [athlete_keys[i] for i in nums if 0 <= i < len(athlete_keys)]
                
                if len(selected_keys) >= 2:
                    visualizer.compare_athletes(selected_keys)
                else:
                    print("❌ Need at least 2 valid athlete numbers")
                    
            except (ValueError, IndexError):
                print("❌ Invalid input format")
                
        elif choice == "4":
            visualizer.generate_data_report()
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 