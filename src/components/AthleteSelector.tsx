import React, { useState, useEffect } from 'react';

interface Athlete {
  id: string;
  name: string;
  sport: string;
  discipline: string;
  video_url: string;
  model_url: string;
}

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

interface AthleteSelectorProps {
  onAthleteSelect: (athlete: AthleteData) => void;
  selectedAthlete?: AthleteData;
}

const AthleteSelector: React.FC<AthleteSelectorProps> = ({
  onAthleteSelect,
  selectedAthlete
}) => {
  const [athletes, setAthletes] = useState<Athlete[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [loadingData, setLoadingData] = useState<string | null>(null);

  useEffect(() => {
    fetchAthletes();
  }, []);

  const fetchAthletes = async () => {
    try {
      const response = await fetch('/api/athletes/list');
      if (!response.ok) {
        throw new Error('Failed to fetch athletes');
      }
      const data = await response.json();
      setAthletes(data.athletes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleAthleteSelect = async (athlete: Athlete) => {
    setLoadingData(athlete.id);
    try {
      const response = await fetch(`/api/athletes/${athlete.id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch athlete data');
      }
      const athleteData: AthleteData = await response.json();
      onAthleteSelect(athleteData);
      setIsOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load athlete data');
    } finally {
      setLoadingData(null);
    }
  };

  const getSportIcon = (sport: string) => {
    switch (sport.toLowerCase()) {
      case 'sprint':
      case 'running':
        return '🏃‍♂️';
      case 'football':
        return '⚽';
      case 'boxing':
        return '🥊';
      case 'rugby':
        return '🏉';
      case 'cricket':
        return '🏏';
      case 'wrestling':
        return '🤼';
      default:
        return '🏅';
    }
  };

  const groupAthletesBySport = (athletes: Athlete[]) => {
    return athletes.reduce((groups, athlete) => {
      const sport = athlete.sport;
      if (!groups[sport]) {
        groups[sport] = [];
      }
      groups[sport].push(athlete);
      return groups;
    }, {} as Record<string, Athlete[]>);
  };

  if (loading) {
    return (
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-lg p-4 border border-gray-700">
        <div className="flex items-center space-x-2">
          <div className="w-5 h-5 text-blue-400 animate-spin">⚡</div>
          <span className="text-gray-300">Loading elite athletes...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 backdrop-blur-sm rounded-lg p-4 border border-red-700">
        <div className="flex items-center space-x-2">
          <span className="text-red-400">Error: {error}</span>
        </div>
      </div>
    );
  }

  const athleteGroups = groupAthletesBySport(athletes);

  return (
    <div className="relative">
      {/* Current Selection */}
      <div 
        className="bg-gray-900/50 backdrop-blur-sm rounded-lg p-4 border border-gray-700 cursor-pointer hover:border-blue-500 transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {selectedAthlete ? (
              <>
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-xl">
                  {getSportIcon(selectedAthlete.sport)}
                </div>
                <div>
                  <h3 className="text-white font-semibold">{selectedAthlete.name}</h3>
                  <p className="text-gray-400 text-sm">{selectedAthlete.sport}</p>
                </div>
              </>
            ) : (
              <>
                <div className="w-10 h-10 text-gray-400 flex items-center justify-center text-2xl">👤</div>
                <div>
                  <h3 className="text-white font-semibold">Select Elite Athlete</h3>
                  <p className="text-gray-400 text-sm">Choose from 13 world-class athletes</p>
                </div>
              </>
            )}
          </div>
          <div className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}>
            ▼
          </div>
        </div>
      </div>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-gray-900/95 backdrop-blur-sm rounded-lg border border-gray-700 shadow-xl z-50 max-h-80 overflow-y-auto">
          {Object.entries(athleteGroups).map(([sport, sportAthletes]) => (
            <div key={sport} className="p-2">
              <div className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-gray-300 border-b border-gray-700">
                <span className="text-lg">{getSportIcon(sport)}</span>
                <span>{sport}</span>
                <span className="text-xs text-gray-500">({sportAthletes.length})</span>
              </div>
              {sportAthletes.map((athlete) => (
                <div
                  key={athlete.id}
                  className="flex items-center justify-between p-3 hover:bg-gray-800/50 cursor-pointer transition-colors"
                  onClick={() => handleAthleteSelect(athlete)}
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white">
                      🏆
                    </div>
                    <div>
                      <p className="text-white font-medium">{athlete.name}</p>
                      <p className="text-gray-400 text-xs">{athlete.discipline}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {loadingData === athlete.id ? (
                      <div className="w-4 h-4 text-blue-400 animate-spin">⚡</div>
                    ) : (
                      <div className="w-4 h-4 text-gray-400">▶</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AthleteSelector; 