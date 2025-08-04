//! # Hardware Integration Error Types

use thiserror::Error;

/// Result type for hardware operations
pub type Result<T> = std::result::Result<T, HardwareError>;

/// Hardware integration error types
#[derive(Error, Debug)]
pub enum HardwareError {
    /// Hardware initialization error
    #[error("Hardware initialization failed: {0}")]
    InitializationError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Communication error with hardware
    #[error("Hardware communication error: {0}")]
    CommunicationError(String),
    
    /// Calibration error
    #[error("Calibration failed: {0}")]
    CalibrationError(String),
    
    /// No hardware readers available
    #[error("No hardware readers available")]
    NoReadersAvailable,
    
    /// Hardware timeout
    #[error("Hardware operation timeout")]
    Timeout,
    
    /// S-entropy engine error
    #[error("S-entropy error: {0}")]
    SEntropyError(#[from] s_entropy_engine::SEntropyError),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}