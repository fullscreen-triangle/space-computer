import {Composition, Folder} from 'remotion';
import React, { useState, useEffect } from 'react';
import { VideoReference } from '../components/VideoReference';
import { BiomechanicalVisualizer } from '../components/BiomechanicalVisualizer';
import AthleteSelector from '../components/AthleteSelector';
import { AIAnalysisPanel } from '../components/AIAnalysisPanel';

interface AthleteData {
  athlete_id: string;
  name: string;
  sport: string;
  discipline: string;
  metadata: {
    fps: number;
    duration: number;
    frame_count: number;
    resolution: string;
  };
  frames: Record<number, any>;
  motion_metrics: any;
  video_path: string;
}

export const EliteAthleteAnalysis: React.FC = () => {
  const [selectedAthlete, setSelectedAthlete] = useState<AthleteData | undefined>();
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Update frame based on time
  useEffect(() => {
    if (selectedAthlete && isPlaying) {
      const fps = selectedAthlete.metadata.fps || 30;
      const totalFrames = selectedAthlete.metadata.frame_count;
      
      const interval = setInterval(() => {
        setCurrentFrame(prev => {
          if (prev >= totalFrames - 1) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 1;
        });
      }, 1000 / fps);

      return () => clearInterval(interval);
    }
  }, [selectedAthlete, isPlaying]);

  const handleAthleteSelect = (athlete: AthleteData) => {
    setSelectedAthlete(athlete);
    setCurrentFrame(0);
    setIsPlaying(false);
    setAnalysis(null);
  };

  const handleAnalysisRequest = async (query?: string) => {
    if (!selectedAthlete) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/athletes/${selectedAthlete.athlete_id}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          frame_number: currentFrame,
          query: query || 'Analyze the biomechanical technique in this frame'
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get analysis');
      }

      const result = await response.json();
      setAnalysis(result.analysis);
    } catch (error) {
      console.error('Analysis error:', error);
      setAnalysis('Failed to generate analysis. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getCurrentFrameData = () => {
    if (!selectedAthlete || !selectedAthlete.frames[currentFrame]) {
      return null;
    }
    return selectedAthlete.frames[currentFrame];
  };

  return (
    <div className="w-full h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white">
      {/* Header */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Space Computer
            </h1>
            <p className="text-gray-300 mt-1">Elite Athlete Biomechanical Analysis</p>
          </div>
          
          <div className="w-80">
            <AthleteSelector
              onAthleteSelect={handleAthleteSelect}
              selectedAthlete={selectedAthlete}
            />
          </div>
        </div>
      </div>

      {!selectedAthlete ? (
        // Welcome Screen
        <div className="flex items-center justify-center h-full">
          <div className="text-center max-w-2xl mx-auto p-8">
            <div className="text-6xl mb-6">🏆</div>
            <h2 className="text-4xl font-bold mb-4">Elite Athlete Analysis Platform</h2>
            <p className="text-xl text-gray-300 mb-8">
              Explore biomechanical data from 13 world-class athletes across multiple sports.
              Select an athlete above to begin analysis.
            </p>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="text-2xl mb-2">🏃‍♂️</div>
                <div className="font-semibold">Sprint Analysis</div>
                <div className="text-gray-400">Usain Bolt, Asafa Powell</div>
              </div>
              <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="text-2xl mb-2">⚽</div>
                <div className="font-semibold">Football Technique</div>
                <div className="text-gray-400">Drogba, Bale, Sterling</div>
              </div>
              <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="text-2xl mb-2">🥊</div>
                <div className="font-semibold">Combat Sports</div>
                <div className="text-gray-400">Boxing, Wrestling</div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        // Main Analysis Interface
        <div className="flex h-full">
          {/* Left Panel - Video Reference */}
          <div className="w-1/2 p-4">
            <div className="bg-gray-800/50 rounded-lg p-4 h-full">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">{selectedAthlete.name} - {selectedAthlete.sport}</h3>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded transition-colors"
                  >
                    {isPlaying ? '⏸️' : '▶️'} {isPlaying ? 'Pause' : 'Play'}
                  </button>
                </div>
              </div>
              
              <VideoReference
                videoUrl={selectedAthlete.video_path}
                currentFrame={currentFrame}
                layout="fullscreen"
                onTimeUpdate={setCurrentFrame}
              />
              
              {/* Frame Controls */}
              <div className="mt-4">
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-300">Frame:</span>
                  <input
                    type="range"
                    min="0"
                    max={selectedAthlete.metadata.frame_count - 1}
                    value={currentFrame}
                    onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-sm text-gray-300">
                    {currentFrame} / {selectedAthlete.metadata.frame_count - 1}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - 3D Visualization and Analysis */}
          <div className="w-1/2 p-4">
            <div className="grid grid-rows-2 gap-4 h-full">
              {/* 3D Biomechanical Visualization */}
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-4">3D Pose Analysis</h3>
                <BiomechanicalVisualizer
                  frameData={getCurrentFrameData()}
                  athleteName={selectedAthlete.name}
                  sport={selectedAthlete.sport}
                />
              </div>

              {/* AI Analysis Panel */}
              <div className="bg-gray-800/50 rounded-lg p-4">
                <AIAnalysisPanel
                  analysis={analysis}
                  loading={loading}
                  onAnalysisRequest={handleAnalysisRequest}
                  athleteName={selectedAthlete.name}
                  currentFrame={currentFrame}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Remotion Composition Registration
export const EliteAthleteCompositions: React.FC = () => {
  return (
    <>
      <Composition
        id="elite-athlete-analysis"
        component={EliteAthleteAnalysis}
        durationInFrames={3600} // 2 minutes at 30fps
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
    </>
  );
}; 