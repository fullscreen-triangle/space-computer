import React, { useState, useEffect } from 'react';

interface VerificationResult {
  understood: boolean;
  confidence: number;
  similarity_score: number;
  verification_time: number;
  error_message?: string;
  verification_id?: string;
}

interface VerificationStatusProps {
  isVerifying: boolean;
  verificationResult?: VerificationResult;
  onRetryVerification?: () => void;
  showDetails?: boolean;
}

const VerificationStatus: React.FC<VerificationStatusProps> = ({
  isVerifying,
  verificationResult,
  onRetryVerification,
  showDetails = false
}) => {
  const [showDetailedView, setShowDetailedView] = useState(showDetails);

  const getStatusColor = () => {
    if (isVerifying) return 'text-blue-600';
    if (!verificationResult) return 'text-gray-400';
    if (verificationResult.understood) return 'text-green-600';
    return 'text-red-600';
  };

  const getStatusIcon = () => {
    if (isVerifying) return <span className="text-lg animate-spin">⏳</span>;
    if (!verificationResult) return <span className="text-lg">👁️</span>;
    if (verificationResult.understood) return <span className="text-lg">✅</span>;
    return <span className="text-lg">⚠️</span>;
  };

  const getStatusMessage = () => {
    if (isVerifying) return 'Verifying AI understanding of pose...';
    if (!verificationResult) return 'Pose understanding not verified';
    if (verificationResult.understood) {
      return `AI understands the pose (${(verificationResult.confidence * 100).toFixed(1)}% confidence)`;
    }
    return 'AI failed to understand the pose - results may be inaccurate';
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.9) return 'Very High';
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.7) return 'Good';
    if (confidence >= 0.6) return 'Moderate';
    return 'Low';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      {/* Main Status Display */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={getStatusColor()}>
            {getStatusIcon()}
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-900">
              Pose Understanding Verification
            </h3>
            <p className={`text-sm ${getStatusColor()}`}>
              {getStatusMessage()}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2">
          {verificationResult && !verificationResult.understood && onRetryVerification && (
                         <button
               onClick={onRetryVerification}
               className="inline-flex items-center px-3 py-1 border border-gray-300 shadow-sm text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
             >
               <span className="mr-1">⚡</span>
               Retry
             </button>
          )}
          
          {verificationResult && (
            <button
              onClick={() => setShowDetailedView(!showDetailedView)}
              className="text-xs text-blue-600 hover:text-blue-800"
            >
              {showDetailedView ? 'Hide Details' : 'Show Details'}
            </button>
          )}
        </div>
      </div>

      {/* Detailed View */}
      {showDetailedView && verificationResult && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-gray-700">Confidence Level:</span>
              <span className={`ml-2 ${getConfidenceColor(verificationResult.confidence)}`}>
                {getConfidenceLevel(verificationResult.confidence)}
              </span>
            </div>
            
            <div>
              <span className="font-medium text-gray-700">Similarity Score:</span>
              <span className="ml-2 text-gray-900">
                {(verificationResult.similarity_score * 100).toFixed(1)}%
              </span>
            </div>
            
            <div>
              <span className="font-medium text-gray-700">Verification Time:</span>
              <span className="ml-2 text-gray-900">
                {verificationResult.verification_time.toFixed(2)}s
              </span>
            </div>
            
            {verificationResult.verification_id && (
              <div>
                <span className="font-medium text-gray-700">Verification ID:</span>
                <span className="ml-2 text-gray-500 font-mono text-xs">
                  {verificationResult.verification_id}
                </span>
              </div>
            )}
          </div>

          {/* Progress Bar for Confidence */}
          <div className="mt-3">
            <div className="flex justify-between text-xs text-gray-600 mb-1">
              <span>Understanding Confidence</span>
              <span>{(verificationResult.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  verificationResult.confidence >= 0.8
                    ? 'bg-green-500'
                    : verificationResult.confidence >= 0.7
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
                }`}
                style={{ width: `${verificationResult.confidence * 100}%` }}
              />
            </div>
          </div>

          {/* Error Message */}
          {verificationResult.error_message && (
            <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
              <strong>Error:</strong> {verificationResult.error_message}
            </div>
          )}

          {/* Explanation */}
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800">
            <p className="font-medium mb-1">What does this mean?</p>
            {verificationResult.understood ? (
              <p>
                The AI has successfully demonstrated understanding of your pose by generating 
                a visual representation that matches the actual pose data. This increases 
                confidence in the analysis results.
              </p>
            ) : (
              <p>
                The AI was unable to accurately visualize your pose, which may indicate 
                limited understanding of the pose data. Analysis results should be 
                interpreted with caution.
              </p>
            )}
          </div>
        </div>
      )}

      {/* Loading State */}
      {isVerifying && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center space-x-2 text-sm text-blue-600">
            <div className="animate-pulse flex space-x-1">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            </div>
            <span>AI is generating pose visualization for verification...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default VerificationStatus; 