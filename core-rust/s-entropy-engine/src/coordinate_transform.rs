//! # Coordinate Transformation
//! 
//! Bidirectional coordinate transformation between S-entropy values and spatial coordinates
//! using the St. Stella constant for mathematical precision.

use crate::st_stella::StStellaConstant;
use crate::error::{SEntropyError, Result};
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

/// S-value coordinate representation
pub type SValueCoordinate = f64;

/// Coordinate transformer for S-entropy ↔ spatial mapping
#[derive(Debug, Clone)]
pub struct CoordinateTransformer {
    /// St. Stella constant for transformations
    st_stella_constant: StStellaConstant,
}

impl CoordinateTransformer {
    /// Creates a new coordinate transformer
    pub fn new(st_stella_constant: StStellaConstant) -> Self {
        Self { st_stella_constant }
    }
    
    /// Transforms S-entropy value to spatial coordinates
    pub fn s_to_spatial(&self, s_value: f64) -> Result<Point3<f64>> {
        if !s_value.is_finite() {
            return Err(SEntropyError::InvalidSValue(s_value));
        }
        
        let scale = self.st_stella_constant.coordinate_scale_factor();
        let phi = self.st_stella_constant.value();
        
        let x = s_value * scale;
        let y = (s_value * phi).sin() * scale;
        let z = (s_value / phi).cos() * scale;
        
        Ok(Point3::new(x, y, z))
    }
    
    /// Transforms spatial coordinates to S-entropy value
    pub fn spatial_to_s(&self, coords: &Point3<f64>) -> Result<f64> {
        let scale = self.st_stella_constant.coordinate_scale_factor();
        Ok(coords.x / scale)
    }
    
    /// Validates if the transformer is operational
    pub fn is_valid(&self) -> bool {
        self.st_stella_constant.is_coherent()
    }
}