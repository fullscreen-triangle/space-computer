//! # Signal Fusion Engine for Multi-Hardware S-Value Reading

use crate::{Result, HardwareError};
use s_entropy_engine::StStellaConstant;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct SignalFusionEngine {
    config: FusionConfig,
}

impl SignalFusionEngine {
    pub fn new(config: FusionConfig) -> Result<Self> {
        Ok(Self { config })
    }
    
    pub fn fuse_readings(&mut self, readings: &[f64]) -> Result<f64> {
        if readings.is_empty() {
            return Err(HardwareError::ConfigurationError("No readings to fuse".into()));
        }
        
        // Simple weighted average for now
        let weights = &self.config.fusion_weights;
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for (i, &reading) in readings.iter().enumerate() {
            let weight = weights.get(i).copied().unwrap_or(1.0);
            weighted_sum += reading * weight;
            total_weight += weight;
        }
        
        Ok(weighted_sum / total_weight)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    pub fusion_algorithm: String,
    pub fusion_weights: Vec<f64>,
    pub st_stella_constant: StStellaConstant,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            fusion_algorithm: "weighted_average".to_string(),
            fusion_weights: vec![1.0, 1.0, 1.0, 1.0],
            st_stella_constant: StStellaConstant::golden_ratio(),
        }
    }
}