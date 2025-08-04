//! # Entropy Endpoints
//! 
//! Implementation of entropy endpoints as oscillation termination points
//! that enable zero-computation navigation through predetermined coordinates.

use crate::st_stella::StStellaConstant;
use crate::error::{SEntropyError, Result};
use serde::{Deserialize, Serialize};
use nalgebra::Vector3;

/// Represents an entropy endpoint in S-entropy space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyEndpoint {
    /// S-entropy value at the endpoint
    pub s_value: f64,
    /// Oscillation amplitude at termination
    pub amplitude: OscillationAmplitude,
    /// Frequency at endpoint
    pub frequency: f64,
    /// Phase at termination
    pub phase: f64,
    /// Endpoint stability measure
    pub stability: f64,
}

impl EntropyEndpoint {
    /// Creates a new entropy endpoint
    pub fn new(s_value: f64, amplitude: OscillationAmplitude) -> Self {
        Self {
            s_value,
            amplitude,
            frequency: amplitude.calculate_endpoint_frequency(),
            phase: amplitude.calculate_endpoint_phase(),
            stability: 1.0,
        }
    }
    
    /// Checks if the endpoint is stable
    pub fn is_stable(&self) -> bool {
        self.stability > 0.95 && 
        self.s_value.is_finite() &&
        self.amplitude.is_valid()
    }
}

/// Oscillation amplitude at entropy endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationAmplitude {
    /// X-component amplitude
    pub x: f64,
    /// Y-component amplitude
    pub y: f64,
    /// Z-component amplitude
    pub z: f64,
}

impl OscillationAmplitude {
    /// Creates a new oscillation amplitude
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    
    /// Calculates the magnitude of the amplitude
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    
    /// Checks if the amplitude is valid
    pub fn is_valid(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    
    /// Calculates endpoint frequency from amplitude
    pub fn calculate_endpoint_frequency(&self) -> f64 {
        self.magnitude() * 2.0 * std::f64::consts::PI
    }
    
    /// Calculates endpoint phase from amplitude
    pub fn calculate_endpoint_phase(&self) -> f64 {
        self.y.atan2(self.x)
    }
    
    /// Converts to vector representation
    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
}