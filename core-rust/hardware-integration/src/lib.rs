//! # Hardware Integration for S-Entropy Framework
//! 
//! This crate provides hardware interfaces for direct S-value measurement
//! from existing hardware systems including LED arrays, MIMO systems, and GPS.
//! 
//! ## Supported Hardware
//! 
//! - **LED Spectrometry Arrays**: Direct S-entropy extraction from light interactions
//! - **MIMO Signal Processing**: S-value measurement through signal coupling analysis  
//! - **GPS Differential Sensing**: Atmospheric S-sensing via propagation delays
//! - **Oscillatory Harvesting**: Hardware oscillation capture from CPUs, WiFi, Bluetooth
//! 
//! ## Core Principle
//! 
//! Instead of requiring specialized sensors, the S-entropy framework enables
//! direct measurement from existing hardware by harvesting their natural
//! oscillatory behavior and signal coupling characteristics.

pub mod led_spectrometry;
pub mod mimo_processing;
pub mod gps_differential;
pub mod oscillatory_harvest;
pub mod signal_fusion;
pub mod error;

pub use led_spectrometry::{LedSpectrometerReader, SpectrometryConfig};
pub use mimo_processing::{MimoSEntropyProcessor, MimoConfig};
pub use gps_differential::{GpsDifferentialReader, GpsConfig};
pub use oscillatory_harvest::{OscillatoryHarvester, HarvestConfig};
pub use signal_fusion::{SignalFusionEngine, FusionConfig};
pub use error::{HardwareError, Result};

use s_entropy_engine::{SEntropyError, SpaceId};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use uuid::Uuid;

/// Core trait for all S-value hardware readers
pub trait SValueReader: Debug + Send + Sync {
    /// Reads S-entropy value for a specific spatial region
    fn read_s_value_for_space(&self, space_id: SpaceId) -> Result<f64>;
    
    /// Returns the reader type identifier
    fn reader_type(&self) -> &str;
    
    /// Checks if the reader is operational
    fn is_operational(&self) -> bool;
    
    /// Returns reader configuration information
    fn configuration(&self) -> ReaderConfig;
    
    /// Performs reader calibration
    fn calibrate(&mut self) -> Result<()>;
    
    /// Returns reader statistics
    fn statistics(&self) -> ReaderStats;
}

/// Configuration information for hardware readers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaderConfig {
    /// Reader unique identifier
    pub id: Uuid,
    /// Reader type name
    pub reader_type: String,
    /// Hardware-specific configuration
    pub hardware_config: serde_json::Value,
    /// Calibration status
    pub calibrated: bool,
    /// Last calibration timestamp
    pub last_calibration: Option<chrono::DateTime<chrono::Utc>>,
}

/// Performance statistics for hardware readers
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReaderStats {
    /// Total number of S-value readings
    pub total_readings: u64,
    /// Successful readings count
    pub successful_readings: u64,
    /// Failed readings count
    pub failed_readings: u64,
    /// Average reading time in microseconds
    pub avg_reading_time_us: u64,
    /// Last reading timestamp
    pub last_reading: Option<chrono::DateTime<chrono::Utc>>,
    /// Reader uptime in seconds
    pub uptime_seconds: u64,
}

impl ReaderStats {
    /// Calculates reading success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_readings == 0 {
            0.0
        } else {
            self.successful_readings as f64 / self.total_readings as f64
        }
    }
}

/// Hardware reader factory for creating specific reader types
pub struct HardwareReaderFactory;

impl HardwareReaderFactory {
    /// Creates a new LED spectrometry reader
    pub fn create_led_reader(config: SpectrometryConfig) -> Result<Box<dyn SValueReader>> {
        Ok(Box::new(LedSpectrometerReader::new(config)?))
    }
    
    /// Creates a new MIMO signal processor
    pub fn create_mimo_reader(config: MimoConfig) -> Result<Box<dyn SValueReader>> {
        Ok(Box::new(MimoSEntropyProcessor::new(config)?))
    }
    
    /// Creates a new GPS differential reader
    pub fn create_gps_reader(config: GpsConfig) -> Result<Box<dyn SValueReader>> {
        Ok(Box::new(GpsDifferentialReader::new(config)?))
    }
    
    /// Creates a new oscillatory harvester
    pub fn create_oscillatory_reader(config: HarvestConfig) -> Result<Box<dyn SValueReader>> {
        Ok(Box::new(OscillatoryHarvester::new(config)?))
    }
    
    /// Creates readers from configuration file
    pub fn create_from_config(config_path: &str) -> Result<Vec<Box<dyn SValueReader>>> {
        let config_content = std::fs::read_to_string(config_path)
            .map_err(|e| HardwareError::ConfigurationError(format!("Failed to read config: {}", e)))?;
        
        let hardware_configs: HardwareConfiguration = serde_json::from_str(&config_content)
            .map_err(|e| HardwareError::ConfigurationError(format!("Invalid config format: {}", e)))?;
        
        let mut readers = Vec::new();
        
        for led_config in hardware_configs.led_configs {
            readers.push(Self::create_led_reader(led_config)?);
        }
        
        for mimo_config in hardware_configs.mimo_configs {
            readers.push(Self::create_mimo_reader(mimo_config)?);
        }
        
        for gps_config in hardware_configs.gps_configs {
            readers.push(Self::create_gps_reader(gps_config)?);
        }
        
        for harvest_config in hardware_configs.harvest_configs {
            readers.push(Self::create_oscillatory_reader(harvest_config)?);
        }
        
        Ok(readers)
    }
}

/// Complete hardware configuration for multiple reader types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfiguration {
    /// LED spectrometry configurations
    pub led_configs: Vec<SpectrometryConfig>,
    /// MIMO processing configurations
    pub mimo_configs: Vec<MimoConfig>,
    /// GPS differential configurations
    pub gps_configs: Vec<GpsConfig>,
    /// Oscillatory harvest configurations
    pub harvest_configs: Vec<HarvestConfig>,
    /// Signal fusion configuration
    pub fusion_config: Option<FusionConfig>,
}

/// Hardware management system for coordinating multiple readers
#[derive(Debug)]
pub struct HardwareManager {
    /// Active hardware readers
    readers: Vec<Box<dyn SValueReader>>,
    /// Signal fusion engine
    fusion_engine: Option<SignalFusionEngine>,
    /// Manager identifier
    id: Uuid,
    /// Manager statistics
    stats: ManagerStats,
}

impl HardwareManager {
    /// Creates a new hardware manager
    pub fn new() -> Self {
        Self {
            readers: Vec::new(),
            fusion_engine: None,
            id: Uuid::new_v4(),
            stats: ManagerStats::default(),
        }
    }
    
    /// Adds a hardware reader to the manager
    pub fn add_reader(&mut self, reader: Box<dyn SValueReader>) {
        self.readers.push(reader);
        self.stats.readers_added += 1;
    }
    
    /// Enables signal fusion for multi-hardware S-value reading
    pub fn enable_fusion(&mut self, config: FusionConfig) -> Result<()> {
        self.fusion_engine = Some(SignalFusionEngine::new(config)?);
        Ok(())
    }
    
    /// Reads S-value from all available hardware and fuses results
    pub async fn read_fused_s_value(&mut self, space_id: SpaceId) -> Result<f64> {
        let mut readings = Vec::new();
        let mut successful_readings = 0;
        
        for reader in &self.readers {
            match reader.read_s_value_for_space(space_id) {
                Ok(value) => {
                    readings.push(value);
                    successful_readings += 1;
                }
                Err(_) => {
                    // Log error but continue with other readers
                    self.stats.failed_readings += 1;
                }
            }
        }
        
        self.stats.total_readings += 1;
        
        if readings.is_empty() {
            return Err(HardwareError::NoReadersAvailable);
        }
        
        // Fuse readings if fusion engine is available
        if let Some(fusion_engine) = &mut self.fusion_engine {
            fusion_engine.fuse_readings(&readings)
        } else {
            // Simple average if no fusion engine
            Ok(readings.iter().sum::<f64>() / readings.len() as f64)
        }
    }
    
    /// Calibrates all hardware readers
    pub async fn calibrate_all(&mut self) -> Result<()> {
        for reader in &mut self.readers {
            reader.calibrate()?;
        }
        self.stats.calibrations_performed += 1;
        Ok(())
    }
    
    /// Returns operational status of all readers
    pub fn operational_status(&self) -> Vec<(String, bool)> {
        self.readers
            .iter()
            .map(|reader| (reader.reader_type().to_string(), reader.is_operational()))
            .collect()
    }
    
    /// Returns manager statistics
    pub fn statistics(&self) -> &ManagerStats {
        &self.stats
    }
}

/// Hardware manager statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManagerStats {
    /// Number of readers added
    pub readers_added: u64,
    /// Total readings performed
    pub total_readings: u64,
    /// Failed readings count
    pub failed_readings: u64,
    /// Calibrations performed
    pub calibrations_performed: u64,
    /// Manager uptime in seconds
    pub uptime_seconds: u64,
}

impl ManagerStats {
    /// Calculates overall success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_readings == 0 {
            0.0
        } else {
            (self.total_readings - self.failed_readings) as f64 / self.total_readings as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hardware_manager_creation() {
        let manager = HardwareManager::new();
        assert_eq!(manager.readers.len(), 0);
        assert!(manager.fusion_engine.is_none());
    }
    
    #[test]
    fn test_reader_stats_success_rate() {
        let mut stats = ReaderStats::default();
        stats.total_readings = 100;
        stats.successful_readings = 95;
        stats.failed_readings = 5;
        
        assert_eq!(stats.success_rate(), 0.95);
    }
    
    #[test]
    fn test_manager_stats_success_rate() {
        let mut stats = ManagerStats::default();
        stats.total_readings = 200;
        stats.failed_readings = 10;
        
        assert_eq!(stats.success_rate(), 0.95);
    }
}