//! # S-Entropy Error Types
//! 
//! Comprehensive error handling for the S-Entropy framework

use thiserror::Error;
use uuid::Uuid;

/// Result type for S-Entropy operations
pub type Result<T> = std::result::Result<T, SEntropyError>;

/// Comprehensive error types for S-Entropy operations
#[derive(Error, Debug)]
pub enum SEntropyError {
    /// Invalid S-entropy value
    #[error("Invalid S-entropy value: {0}")]
    InvalidSValue(f64),
    
    /// No baseline established for space
    #[error("No baseline S-entropy value for space: {0}")]
    NoBaseline(Uuid),
    
    /// Hardware failure
    #[error("Hardware failure: {0}")]
    HardwareFailure(String),
    
    /// Matrix inversion error
    #[error("Matrix inversion error: {0}")]
    MatrixInversion(String),
    
    /// Insufficient data for operation
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    /// Coordinate transformation error
    #[error("Coordinate transformation failed: {0}")]
    CoordinateTransformation(String),
    
    /// St. Stella constant coherence error
    #[error("St. Stella constant is not coherent")]
    IncoherentConstant,
    
    /// Navigation error
    #[error("Navigation failed: {0}")]
    NavigationError(String),
    
    /// Cache operation error
    #[error("Cache operation failed: {0}")]
    CacheError(String),
    
    /// Generic computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
}