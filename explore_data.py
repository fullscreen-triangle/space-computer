#!/usr/bin/env python3
"""
Biomechanical Data Explorer
==========================
Interactive exploration tool for the datasources folder
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set up dark theme for better visualization
plt.style.use('dark_background')
sns.set_palette("husl")

class DataExplorer:
    def __init__(self):
        self.data_path = Path("datasources")
        self.athletes = {}
        
    def scan_data(self):
        """Scan and catalog all available data"""
        print("🔍 Scanning datasources folder...")
        
        # Models directory - pose data
        models_dir = self.data_path / "models" 
        posture_dir = self.data_path / "posture"
        video_dir = self.data_path / "annotated"
        
        print(f"📁 Models: {len(list(models_dir.glob('*.json')))} files")
        print(f"📁 Posture: {len(list(posture_dir.glob('*.json')))} files")  
        print(f"📁 Videos: {len(list(video_dir.glob('*.mp4')))} files")
        
        # Parse athlete data
        for pose_file in models_dir.glob("*_pose_data.json"):
            athlete_name = pose_file.stem.replace("_pose_data", "")
            
            self.athletes[athlete_name] = {
                "name": self._clean_name(athlete_name),
                "pose_file": pose_file,
                "sport": self._detect_sport(athlete_name),
                "video_file": video_dir / f"{athlete_name}_annotated.mp4"
            }
            
        print(f"\n✅ Found {len(self.athletes)} athletes")
        return self.athletes
        
    def _clean_name(self, name):
        """Clean up athlete names"""
        return name.replace("-", " ").replace("_", " ").title()
        
    def _detect_sport(self, name):
        """Detect sport from athlete name"""
        sports = {
            "bolt": "Sprinting", "powell": "Sprinting", "beijing": "Athletics",
            "drogba": "Football", "koroibete": "Rugby", "lomu": "Rugby", 
            "chisora": "Boxing", "pound": "Boxing", "struggle": "Wrestling",
            "boundary": "Cricket", "hezvo": "Track"
        }
        
        for key, sport in sports.items():
            if key in name.lower():
                return sport
        return "General"
        
    def load_pose_data(self, athlete_name):
        """Load pose data for specific athlete"""
        if athlete_name not in self.athletes:
            print(f"❌ Athlete {athlete_name} not found")
            return None
            
        athlete = self.athletes[athlete_name]
        
        try:
            with open(athlete["pose_file"], 'r') as f:
                data = json.load(f)
                athlete["pose_data"] = data
                print(f"✅ Loaded pose data for {athlete['name']}")
                return data
        except Exception as e:
            print(f"❌ Error loading {athlete['name']}: {e}")
            return None
            
    def analyze_pose_quality(self, athlete_name):
        """Analyze pose detection quality"""
        data = self.load_pose_data(athlete_name)
        if not data:
            return
            
        video_info = data["video_info"]
        frames = data["pose_data"]
        
        print(f"\n📊 POSE ANALYSIS: {self.athletes[athlete_name]['name']}")
        print(f"   Video: {video_info['width']}x{video_info['height']} @ {video_info['fps']}fps")
        print(f"   Duration: {video_info['total_frames']/video_info['fps']:.1f}s")
        print(f"   Total Frames: {video_info['total_frames']}")
        
        # Analyze pose detection
        pose_counts = []
        confidence_scores = []
        
        for frame in frames:
            pose_count = len(frame["poses"])
            pose_counts.append(pose_count)
            
            if pose_count > 0:
                # Get average confidence
                landmarks = frame["poses"][0]["landmarks"]
                confidences = [lm["visibility"] for lm in landmarks if lm["visibility"] > 0.5]
                if confidences:
                    confidence_scores.append(np.mean(confidences))
                    
        # Statistics
        detection_rate = sum(1 for x in pose_counts if x > 0) / len(pose_counts)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        print(f"   Detection Rate: {detection_rate:.1%}")
        print(f"   Avg Confidence: {avg_confidence:.3f}")
        print(f"   Frames with poses: {len(confidence_scores)}/{len(frames)}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Pose Quality Analysis - {self.athletes[athlete_name]['name']}", fontsize=16)
        
        # Detection over time
        axes[0,0].plot(pose_counts, 'cyan', linewidth=2)
        axes[0,0].set_title("Pose Detection Over Time")
        axes[0,0].set_xlabel("Frame")
        axes[0,0].set_ylabel("Poses Detected")
        axes[0,0].grid(True, alpha=0.3)
        
        # Confidence over time
        if confidence_scores:
            frame_indices = [i for i, count in enumerate(pose_counts) if count > 0]
            axes[0,1].plot(frame_indices, confidence_scores, 'lime', linewidth=2)
            axes[0,1].set_title("Pose Confidence Over Time")
            axes[0,1].set_xlabel("Frame")
            axes[0,1].set_ylabel("Confidence Score")
            axes[0,1].set_ylim(0, 1)
            axes[0,1].grid(True, alpha=0.3)
        
        # Confidence histogram
        if confidence_scores:
            axes[1,0].hist(confidence_scores, bins=20, color='orange', alpha=0.7)
            axes[1,0].set_title("Confidence Distribution")
            axes[1,0].set_xlabel("Confidence Score")
            axes[1,0].set_ylabel("Frequency")
        
        # Sample pose
        self._plot_sample_pose(axes[1,1], frames)
        
        plt.tight_layout()
        plt.show()
        
        return {
            "detection_rate": detection_rate,
            "avg_confidence": avg_confidence,
            "total_frames": len(frames),
            "frames_with_poses": len(confidence_scores)
        }
        
    def _plot_sample_pose(self, ax, frames):
        """Plot a sample pose visualization"""
        # Find a frame with good pose data
        for frame in frames[len(frames)//2:]:
            if frame["poses"]:
                landmarks = frame["poses"][0]["landmarks"]
                
                # Extract coordinates for visible landmarks
                x_coords = []
                y_coords = []
                
                for lm in landmarks:
                    if lm["visibility"] > 0.5:
                        x_coords.append(lm["x"])
                        y_coords.append(1 - lm["y"])  # Flip Y for proper orientation
                        
                if len(x_coords) > 10:  # Ensure we have enough points
                    ax.scatter(x_coords, y_coords, c='red', s=30, alpha=0.8)
                    ax.set_title("Sample Pose Keypoints")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1) 
                    ax.set_aspect('equal')
                    break
                    
    def compare_athletes(self, athlete_names):
        """Compare multiple athletes"""
        if len(athlete_names) < 2:
            print("❌ Need at least 2 athletes to compare")
            return
            
        comparison_data = {}
        
        for name in athlete_names:
            stats = self.analyze_pose_quality(name)
            if stats:
                comparison_data[self.athletes[name]['name']] = stats
                
        if len(comparison_data) < 2:
            print("❌ Not enough valid data for comparison")
            return
            
        # Create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Athlete Comparison", fontsize=16)
        
        names = list(comparison_data.keys())
        detection_rates = [comparison_data[name]['detection_rate'] for name in names]
        confidence_scores = [comparison_data[name]['avg_confidence'] for name in names]
        frame_counts = [comparison_data[name]['total_frames'] for name in names]
        
        # Detection rates
        bars1 = axes[0].bar(names, detection_rates, color='cyan', alpha=0.7)
        axes[0].set_title("Pose Detection Rate")
        axes[0].set_ylabel("Detection Rate")
        axes[0].set_ylim(0, 1)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # Confidence scores
        bars2 = axes[1].bar(names, confidence_scores, color='lime', alpha=0.7)
        axes[1].set_title("Average Confidence")
        axes[1].set_ylabel("Confidence Score")
        axes[1].set_ylim(0, 1)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        # Frame counts
        bars3 = axes[2].bar(names, frame_counts, color='orange', alpha=0.7)
        axes[2].set_title("Total Frames")
        axes[2].set_ylabel("Frame Count")
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bars, values in [(bars1, detection_rates), (bars2, confidence_scores), (bars3, frame_counts)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if isinstance(value, float) and value < 1:
                    label = f'{value:.2f}'
                else:
                    label = f'{int(value)}'
                axes[bars.get_label()].text(bar.get_x() + bar.get_width()/2., height,
                                          label, ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return comparison_data
        
    def list_athletes(self):
        """Display all available athletes"""
        print("\n🏆 AVAILABLE ATHLETES")
        print("-" * 50)
        
        for i, (key, athlete) in enumerate(self.athletes.items(), 1):
            video_exists = "✅" if athlete["video_file"].exists() else "❌"
            print(f"{i:2d}. {athlete['name']:<20} | {athlete['sport']:<12} | Video: {video_exists}")
            
    def show_dataset_overview(self):
        """Show comprehensive dataset overview"""
        print("\n📊 DATASET OVERVIEW")
        print("=" * 60)
        
        # Sports distribution
        sports = {}
        total_size = 0
        
        for athlete in self.athletes.values():
            sport = athlete['sport']
            sports[sport] = sports.get(sport, 0) + 1
            
            # Check file sizes
            if athlete['pose_file'].exists():
                total_size += athlete['pose_file'].stat().st_size
                
        print(f"Total Athletes: {len(self.athletes)}")
        print(f"Total Data Size: {total_size / (1024*1024):.1f} MB")
        print(f"Sports Represented: {len(sports)}")
        
        for sport, count in sorted(sports.items()):
            print(f"  {sport}: {count} athletes")
            
        # Visualize sports distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart of sports
        ax1.pie(sports.values(), labels=sports.keys(), autopct='%1.1f%%', startangle=90)
        ax1.set_title("Sports Distribution")
        
        # Bar chart with file sizes
        athlete_names = [athlete['name'][:10] for athlete in self.athletes.values()]
        file_sizes = [athlete['pose_file'].stat().st_size / (1024*1024) 
                     if athlete['pose_file'].exists() else 0 
                     for athlete in self.athletes.values()]
        
        bars = ax2.bar(range(len(athlete_names)), file_sizes, color='skyblue', alpha=0.7)
        ax2.set_title("Data File Sizes (MB)")
        ax2.set_xlabel("Athletes")
        ax2.set_ylabel("File Size (MB)")
        ax2.set_xticks(range(len(athlete_names)))
        ax2.set_xticklabels(athlete_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main interactive loop"""
    explorer = DataExplorer()
    
    # Initial scan
    athletes = explorer.scan_data()
    explorer.show_dataset_overview()
    
    while True:
        print("\n" + "="*50)
        print("🏃‍♂️ BIOMECHANICAL DATA EXPLORER")
        print("="*50)
        print("1. List all athletes")
        print("2. Analyze individual athlete")
        print("3. Compare multiple athletes")
        print("4. Show dataset overview")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            explorer.list_athletes()
            
        elif choice == "2":
            explorer.list_athletes()
            try:
                num = int(input("\nSelect athlete number: ")) - 1
                athlete_keys = list(athletes.keys())
                if 0 <= num < len(athlete_keys):
                    selected = athlete_keys[num]
                    explorer.analyze_pose_quality(selected)
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")
                
        elif choice == "3":
            explorer.list_athletes()
            try:
                nums = input("\nSelect athlete numbers (comma-separated): ").split(",")
                athlete_keys = list(athletes.keys())
                selected = [athlete_keys[int(n.strip())-1] for n in nums 
                           if 0 <= int(n.strip())-1 < len(athlete_keys)]
                if len(selected) >= 2:
                    explorer.compare_athletes(selected)
                else:
                    print("❌ Need at least 2 valid selections")
            except (ValueError, IndexError):
                print("❌ Invalid input")
                
        elif choice == "4":
            explorer.show_dataset_overview()
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 