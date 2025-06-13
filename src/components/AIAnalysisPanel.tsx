import React, { useState } from 'react';

interface AIAnalysisPanelProps {
  analysis: string | null;
  loading: boolean;
  onAnalysisRequest: (query?: string) => Promise<void>;
  athleteName: string;
  currentFrame: number;
}

export const AIAnalysisPanel: React.FC<AIAnalysisPanelProps> = ({
  analysis,
  loading,
  onAnalysisRequest,
  athleteName,
  currentFrame
}) => {
  const [customQuery, setCustomQuery] = useState('');

  const handleAnalyze = () => {
    onAnalysisRequest(customQuery || undefined);
  };

  const predefinedQueries = [
    'Analyze the biomechanical technique in this frame',
    'What are the key strengths in this movement pattern?',
    'Identify potential areas for improvement',
    'Compare this technique to optimal form',
    'Analyze joint angles and force distribution'
  ];

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">AI Biomechanical Analysis</h3>
        <div className="text-sm text-gray-400">
          Frame {currentFrame} - {athleteName}
        </div>
      </div>

      {/* Query Input */}
      <div className="mb-4">
        <input
          type="text"
          value={customQuery}
          onChange={(e) => setCustomQuery(e.target.value)}
          placeholder="Ask a specific question about the technique..."
          className="w-full bg-gray-700 text-white px-3 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
          aria-label="Analysis query input"
        />
        <div className="flex flex-wrap gap-2 mt-2">
          {predefinedQueries.slice(0, 3).map((query, index) => (
            <button
              key={index}
              onClick={() => setCustomQuery(query)}
              className="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded transition-colors"
            >
              {query}
            </button>
          ))}
        </div>
      </div>

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={loading}
        className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded transition-colors mb-4"
      >
        {loading ? (
          <span className="flex items-center space-x-2">
            <div className="animate-spin">⚡</div>
            <span>Analyzing...</span>
          </span>
        ) : (
          'Analyze Movement'
        )}
      </button>

      {/* Analysis Results */}
      <div className="flex-1 overflow-y-auto">
        {analysis ? (
          <div className="bg-gray-700/50 rounded-lg p-4">
            <h4 className="font-semibold mb-2 text-blue-400">Analysis Result:</h4>
            <div className="text-sm text-gray-200 whitespace-pre-wrap leading-relaxed">
              {analysis}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <div className="text-4xl mb-2">🧠</div>
              <p>Click "Analyze Movement" to get AI insights</p>
              <p className="text-xs mt-1">Powered by specialized biomechanical LLM</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 