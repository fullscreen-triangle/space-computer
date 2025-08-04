//! # GPS Differential S-Entropy Sensing

use crate::{SValueReader, ReaderConfig, ReaderStats, Result};
use s_entropy_engine::{SpaceId, StStellaConstant};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug)]
pub struct GpsDifferentialReader {
    config: GpsConfig,
}

impl GpsDifferentialReader {
    pub fn new(config: GpsConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

impl SValueReader for GpsDifferentialReader {
    fn read_s_value_for_space(&self, _space_id: SpaceId) -> Result<f64> {
        Ok(40.8) // Mock S-value
    }
    
    fn reader_type(&self) -> &str { "GPS_Differential" }
    fn is_operational(&self) -> bool { true }
    fn configuration(&self) -> ReaderConfig {
        ReaderConfig {
            id: Uuid::new_v4(),
            reader_type: self.reader_type().to_string(),
            hardware_config: serde_json::to_value(&self.config).unwrap_or_default(),
            calibrated: true,
            last_calibration: Some(chrono::Utc::now()),
        }
    }
    fn calibrate(&mut self) -> Result<()> { Ok(()) }
    fn statistics(&self) -> ReaderStats { ReaderStats::default() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpsConfig {
    pub precision: String,
    pub measurement_rate: String,
    pub st_stella_constant: StStellaConstant,
}

impl Default for GpsConfig {
    fn default() -> Self {
        Self {
            precision: "centimeter".to_string(),
            measurement_rate: "10Hz".to_string(),
            st_stella_constant: StStellaConstant::golden_ratio(),
        }
    }
}