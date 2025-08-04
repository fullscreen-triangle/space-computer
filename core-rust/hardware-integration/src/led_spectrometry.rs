//! # LED Spectrometry S-Value Reader
//! 
//! Revolutionary S-entropy extraction from LED array interactions with gas molecules.
//! This module implements the breakthrough discovery that LED light interactions
//! with atmospheric gas can be analyzed to extract S-entropy values directly.

use crate::{SValueReader, ReaderConfig, ReaderStats, Result, HardwareError};
use s_entropy_engine::{SpaceId, StStellaConstant};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;
use nalgebra::Vector3;
use ndarray::Array1;
use rustfft::{FftPlanner, num_complex::Complex};

/// LED Spectrometry Reader for direct S-entropy measurement from light interactions
#[derive(Debug)]
pub struct LedSpectrometerReader {
    /// Reader configuration
    config: SpectrometryConfig,
    /// LED controllers for different wavelengths
    led_controllers: Vec<LedController>,
    /// Spectrometry analysis configuration
    spectrometry_config: SpectrometerConfig,
    /// S-value extraction algorithm
    s_extraction_algorithm: SExtractionAlgorithm,
    /// Reader statistics
    stats: ReaderStats,
    /// Calibration data
    calibration_data: Option<CalibrationData>,
    /// Last reading cache
    reading_cache: HashMap<SpaceId, (f64, Instant)>,
}

impl LedSpectrometerReader {
    /// Creates a new LED spectrometer reader
    pub fn new(config: SpectrometryConfig) -> Result<Self> {
        let led_controllers = Self::initialize_led_controllers(&config)?;
        let spectrometry_config = SpectrometerConfig::from_led_config(&config);
        let s_extraction_algorithm = SExtractionAlgorithm::new(config.st_stella_constant.clone())?;
        
        Ok(Self {
            config,
            led_controllers,
            spectrometry_config,
            s_extraction_algorithm,
            stats: ReaderStats::default(),
            calibration_data: None,
            reading_cache: HashMap::new(),
        })
    }
    
    /// Initializes LED controllers based on configuration
    fn initialize_led_controllers(config: &SpectrometryConfig) -> Result<Vec<LedController>> {
        let mut controllers = Vec::new();
        
        for led_spec in &config.led_specifications {
            let controller = LedController::new(led_spec.clone())?;
            controllers.push(controller);
        }
        
        if controllers.is_empty() {
            return Err(HardwareError::InitializationError("No LED controllers configured".into()));
        }
        
        Ok(controllers)
    }
    
    /// Captures LED spectrum for a spatial region
    fn capture_led_spectrum(&mut self, space_region: SpaceRegion) -> Result<SpectrumData> {
        let start_time = Instant::now();
        let mut spectrum_data = SpectrumData::new();
        
        // Activate LEDs sequentially and capture response
        for controller in &mut self.led_controllers {
            // Turn on LED at specific wavelength
            controller.activate()?;
            
            // Wait for gas interaction stabilization
            std::thread::sleep(Duration::from_millis(self.config.stabilization_time_ms));
            
            // Capture spectral response
            let wavelength_response = self.capture_wavelength_response(controller, &space_region)?;
            spectrum_data.add_wavelength_data(controller.wavelength(), wavelength_response);
            
            // Turn off LED
            controller.deactivate()?;
        }
        
        self.stats.avg_reading_time_us = start_time.elapsed().as_micros() as u64;
        self.stats.total_readings += 1;
        
        Ok(spectrum_data)
    }
    
    /// Captures response for a specific wavelength
    fn capture_wavelength_response(&self, controller: &LedController, space_region: &SpaceRegion) -> Result<Array1<f64>> {
        // Simulate spectral response capture
        // In real implementation, this would interface with photodetectors
        let mut response = Array1::zeros(self.spectrometry_config.resolution);
        
        let base_wavelength = controller.wavelength();
        let intensity = controller.current_intensity();
        
        // Generate realistic spectral response based on gas interactions
        for (i, value) in response.iter_mut().enumerate() {
            let wavelength = base_wavelength + (i as f64 * self.spectrometry_config.wavelength_step);
            
            // Simulate gas absorption and scattering effects
            let gas_absorption = self.calculate_gas_absorption(wavelength, space_region);
            let scattering_effect = self.calculate_scattering_effect(wavelength, space_region);
            
            *value = intensity * (1.0 - gas_absorption) * scattering_effect + 
                     self.generate_noise();
        }
        
        Ok(response)
    }
    
    /// Calculates gas absorption at specific wavelength
    fn calculate_gas_absorption(&self, wavelength: f64, space_region: &SpaceRegion) -> f64 {
        // Simplified gas absorption model
        // Real implementation would use molecular absorption databases
        let absorption_coefficient = 0.001; // Base absorption
        let density_factor = space_region.estimated_gas_density;
        
        absorption_coefficient * density_factor * (wavelength / 500.0).exp()
    }
    
    /// Calculates scattering effects
    fn calculate_scattering_effect(&self, wavelength: f64, space_region: &SpaceRegion) -> f64 {
        // Rayleigh scattering model (1/λ^4 dependence)
        let scattering_base = 0.1;
        let wavelength_factor = (400.0 / wavelength).powi(4);
        
        1.0 - scattering_base * wavelength_factor * space_region.estimated_gas_density
    }
    
    /// Generates realistic noise for spectral measurements
    fn generate_noise(&self) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(-0.01..0.01) // 1% noise level
    }
}

impl SValueReader for LedSpectrometerReader {
    /// Reads S-entropy value through LED spectrometry analysis
    fn read_s_value_for_space(&self, space_id: SpaceId) -> Result<f64> {
        // Check cache for recent readings
        if let Some((cached_value, timestamp)) = self.reading_cache.get(&space_id) {
            if timestamp.elapsed() < Duration::from_millis(self.config.cache_duration_ms) {
                return Ok(*cached_value);
            }
        }
        
        let space_region = SpaceRegion::from_space_id(space_id);
        let mut reader = self.clone(); // Temporary for mutable operations
        
        // Capture LED spectrum
        let spectrum_data = reader.capture_led_spectrum(space_region)?;
        
        // Extract S-entropy value from spectrum
        let s_value = self.s_extraction_algorithm.extract_s_from_spectrum(spectrum_data)?;
        
        // Update cache
        reader.reading_cache.insert(space_id, (s_value, Instant::now()));
        
        Ok(s_value)
    }
    
    fn reader_type(&self) -> &str {
        "LED_Spectrometry"
    }
    
    fn is_operational(&self) -> bool {
        self.calibration_data.is_some() && 
        self.led_controllers.iter().all(|c| c.is_operational()) &&
        self.config.st_stella_constant.is_coherent()
    }
    
    fn configuration(&self) -> ReaderConfig {
        ReaderConfig {
            id: Uuid::new_v4(),
            reader_type: self.reader_type().to_string(),
            hardware_config: serde_json::to_value(&self.config).unwrap_or_default(),
            calibrated: self.calibration_data.is_some(),
            last_calibration: self.calibration_data.as_ref().map(|d| d.calibration_timestamp),
        }
    }
    
    fn calibrate(&mut self) -> Result<()> {
        // Perform LED calibration sequence
        let mut calibration_data = CalibrationData::new();
        
        for controller in &mut self.led_controllers {
            let calibration_spectrum = self.capture_calibration_spectrum(controller)?;
            calibration_data.add_led_calibration(controller.wavelength(), calibration_spectrum);
        }
        
        self.calibration_data = Some(calibration_data);
        Ok(())
    }
    
    fn statistics(&self) -> ReaderStats {
        self.stats.clone()
    }
}

impl Clone for LedSpectrometerReader {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            led_controllers: self.led_controllers.clone(),
            spectrometry_config: self.spectrometry_config.clone(),
            s_extraction_algorithm: self.s_extraction_algorithm.clone(),
            stats: self.stats.clone(),
            calibration_data: self.calibration_data.clone(),
            reading_cache: HashMap::new(), // Don't clone cache
        }
    }
}

/// Configuration for LED spectrometry reader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrometryConfig {
    /// St. Stella constant for S-entropy calculations
    pub st_stella_constant: StStellaConstant,
    /// LED specifications
    pub led_specifications: Vec<LedSpecification>,
    /// Measurement resolution
    pub resolution: usize,
    /// Wavelength range in nanometers
    pub wavelength_range: (f64, f64),
    /// Stabilization time in milliseconds
    pub stabilization_time_ms: u64,
    /// Cache duration in milliseconds
    pub cache_duration_ms: u64,
    /// S-extraction algorithm type
    pub extraction_algorithm: String,
}

impl Default for SpectrometryConfig {
    fn default() -> Self {
        Self {
            st_stella_constant: StStellaConstant::golden_ratio(),
            led_specifications: vec![
                LedSpecification::rgb_red(),
                LedSpecification::rgb_green(),
                LedSpecification::rgb_blue(),
            ],
            resolution: 1000,
            wavelength_range: (380.0, 700.0),
            stabilization_time_ms: 10,
            cache_duration_ms: 100,
            extraction_algorithm: "spectral_signature_analysis".to_string(),
        }
    }
}

/// LED specification for different wavelengths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedSpecification {
    /// LED wavelength in nanometers
    pub wavelength: f64,
    /// Maximum intensity
    pub max_intensity: f64,
    /// LED type (RGB, IR, UV, etc.)
    pub led_type: String,
    /// Hardware interface (GPIO pin, I2C address, etc.)
    pub hardware_interface: String,
}

impl LedSpecification {
    /// Creates RGB red LED specification
    pub fn rgb_red() -> Self {
        Self {
            wavelength: 660.0,
            max_intensity: 1.0,
            led_type: "RGB_Red".to_string(),
            hardware_interface: "GPIO_18".to_string(),
        }
    }
    
    /// Creates RGB green LED specification
    pub fn rgb_green() -> Self {
        Self {
            wavelength: 525.0,
            max_intensity: 1.0,
            led_type: "RGB_Green".to_string(),
            hardware_interface: "GPIO_19".to_string(),
        }
    }
    
    /// Creates RGB blue LED specification
    pub fn rgb_blue() -> Self {
        Self {
            wavelength: 470.0,
            max_intensity: 1.0,
            led_type: "RGB_Blue".to_string(),
            hardware_interface: "GPIO_20".to_string(),
        }
    }
}

/// LED controller for individual wavelengths
#[derive(Debug, Clone)]
pub struct LedController {
    /// LED specification
    spec: LedSpecification,
    /// Current intensity (0.0 to 1.0)
    current_intensity: f64,
    /// Operational status
    operational: bool,
}

impl LedController {
    /// Creates a new LED controller
    pub fn new(spec: LedSpecification) -> Result<Self> {
        Ok(Self {
            spec,
            current_intensity: 0.0,
            operational: true, // Assume operational for simulation
        })
    }
    
    /// Activates LED at full intensity
    pub fn activate(&mut self) -> Result<()> {
        self.current_intensity = self.spec.max_intensity;
        // In real implementation, would control actual hardware
        Ok(())
    }
    
    /// Deactivates LED
    pub fn deactivate(&mut self) -> Result<()> {
        self.current_intensity = 0.0;
        Ok(())
    }
    
    /// Returns LED wavelength
    pub fn wavelength(&self) -> f64 {
        self.spec.wavelength
    }
    
    /// Returns current intensity
    pub fn current_intensity(&self) -> f64 {
        self.current_intensity
    }
    
    /// Checks if controller is operational
    pub fn is_operational(&self) -> bool {
        self.operational
    }
}

/// Spectrometer configuration derived from LED configuration
#[derive(Debug, Clone)]
pub struct SpectrometerConfig {
    /// Spectral resolution
    pub resolution: usize,
    /// Wavelength step size
    pub wavelength_step: f64,
    /// Integration time in milliseconds
    pub integration_time_ms: u64,
}

impl SpectrometerConfig {
    /// Creates spectrometer config from LED configuration
    pub fn from_led_config(led_config: &SpectrometryConfig) -> Self {
        let wavelength_range = led_config.wavelength_range.1 - led_config.wavelength_range.0;
        let wavelength_step = wavelength_range / led_config.resolution as f64;
        
        Self {
            resolution: led_config.resolution,
            wavelength_step,
            integration_time_ms: led_config.stabilization_time_ms,
        }
    }
}

/// S-entropy extraction algorithm for spectral data
#[derive(Debug, Clone)]
pub struct SExtractionAlgorithm {
    /// St. Stella constant for calculations
    st_stella_constant: StStellaConstant,
    /// Algorithm type
    algorithm_type: AlgorithmType,
}

impl SExtractionAlgorithm {
    /// Creates a new S-extraction algorithm
    pub fn new(st_stella_constant: StStellaConstant) -> Result<Self> {
        Ok(Self {
            st_stella_constant,
            algorithm_type: AlgorithmType::SpectralSignatureAnalysis,
        })
    }
    
    /// Extracts S-entropy value from spectrum data
    pub fn extract_s_from_spectrum(&self, spectrum_data: SpectrumData) -> Result<f64> {
        match self.algorithm_type {
            AlgorithmType::SpectralSignatureAnalysis => {
                self.spectral_signature_analysis(spectrum_data)
            },
            AlgorithmType::FourierTransformAnalysis => {
                self.fourier_transform_analysis(spectrum_data)
            },
            AlgorithmType::WaveletAnalysis => {
                self.wavelet_analysis(spectrum_data)
            },
        }
    }
    
    /// Spectral signature analysis algorithm
    fn spectral_signature_analysis(&self, spectrum_data: SpectrumData) -> Result<f64> {
        // Calculate spectral signature characteristics
        let spectral_moments = spectrum_data.calculate_moments();
        let peak_analysis = spectrum_data.analyze_peaks();
        let absorption_features = spectrum_data.identify_absorption_features();
        
        // Transform to S-entropy using St. Stella constant
        let signature_value = spectral_moments.weighted_sum() + 
                            peak_analysis.dominant_peak_intensity + 
                            absorption_features.total_absorption;
        
        let s_value = self.st_stella_constant.transform_entropy(signature_value);
        Ok(s_value)
    }
    
    /// Fourier transform analysis algorithm
    fn fourier_transform_analysis(&self, spectrum_data: SpectrumData) -> Result<f64> {
        // Perform FFT on spectral data
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(spectrum_data.data.len());
        
        let mut buffer: Vec<Complex<f64>> = spectrum_data.data
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        fft.process(&mut buffer);
        
        // Extract frequency domain features
        let power_spectrum: Vec<f64> = buffer.iter()
            .map(|c| c.norm_sqr())
            .collect();
        
        let dominant_frequency = power_spectrum.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as f64)
            .unwrap_or(0.0);
        
        let s_value = self.st_stella_constant.transform_entropy(dominant_frequency);
        Ok(s_value)
    }
    
    /// Wavelet analysis algorithm
    fn wavelet_analysis(&self, spectrum_data: SpectrumData) -> Result<f64> {
        // Simplified wavelet analysis
        // Real implementation would use proper wavelet transform libraries
        let data_variance = spectrum_data.calculate_variance();
        let data_skewness = spectrum_data.calculate_skewness();
        
        let wavelet_signature = data_variance * data_skewness.abs();
        let s_value = self.st_stella_constant.transform_entropy(wavelet_signature);
        Ok(s_value)
    }
}

/// Types of S-extraction algorithms
#[derive(Debug, Clone)]
pub enum AlgorithmType {
    SpectralSignatureAnalysis,
    FourierTransformAnalysis,
    WaveletAnalysis,
}

/// Spectral data from LED measurements
#[derive(Debug, Clone)]
pub struct SpectrumData {
    /// Wavelength data points
    pub wavelengths: Vec<f64>,
    /// Intensity data
    pub data: Vec<f64>,
    /// Metadata
    pub metadata: SpectrumMetadata,
}

impl SpectrumData {
    /// Creates new empty spectrum data
    pub fn new() -> Self {
        Self {
            wavelengths: Vec::new(),
            data: Vec::new(),
            metadata: SpectrumMetadata::default(),
        }
    }
    
    /// Adds wavelength data point
    pub fn add_wavelength_data(&mut self, wavelength: f64, data: Array1<f64>) {
        self.wavelengths.push(wavelength);
        self.data.extend(data.iter());
    }
    
    /// Calculates spectral moments for signature analysis
    pub fn calculate_moments(&self) -> SpectralMoments {
        if self.data.is_empty() {
            return SpectralMoments::default();
        }
        
        let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
        let variance = self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.data.len() as f64;
        
        SpectralMoments {
            mean,
            variance,
            skewness: self.calculate_skewness(),
            kurtosis: self.calculate_kurtosis(),
        }
    }
    
    /// Calculates variance
    pub fn calculate_variance(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        
        let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
        self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.data.len() as f64
    }
    
    /// Calculates skewness
    pub fn calculate_skewness(&self) -> f64 {
        if self.data.len() < 3 { return 0.0; }
        
        let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
        let variance = self.calculate_variance();
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 { return 0.0; }
        
        let skewness = self.data.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / self.data.len() as f64;
        
        skewness
    }
    
    /// Calculates kurtosis
    pub fn calculate_kurtosis(&self) -> f64 {
        if self.data.len() < 4 { return 0.0; }
        
        let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
        let variance = self.calculate_variance();
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 { return 0.0; }
        
        let kurtosis = self.data.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / self.data.len() as f64;
        
        kurtosis - 3.0 // Excess kurtosis
    }
    
    /// Analyzes spectral peaks
    pub fn analyze_peaks(&self) -> PeakAnalysis {
        if self.data.is_empty() {
            return PeakAnalysis::default();
        }
        
        let max_intensity = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_intensity = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        
        PeakAnalysis {
            dominant_peak_intensity: max_intensity,
            peak_count: self.count_peaks(),
            peak_width_average: self.calculate_average_peak_width(),
        }
    }
    
    /// Identifies absorption features
    pub fn identify_absorption_features(&self) -> AbsorptionFeatures {
        let total_absorption = self.data.iter()
            .enumerate()
            .filter(|(_, &intensity)| intensity < 0.5) // Absorption threshold
            .map(|(_, &intensity)| 1.0 - intensity)
            .sum::<f64>();
        
        AbsorptionFeatures {
            total_absorption,
            absorption_lines: self.count_absorption_lines(),
        }
    }
    
    /// Counts spectral peaks
    fn count_peaks(&self) -> usize {
        let mut peaks = 0;
        for i in 1..self.data.len().saturating_sub(1) {
            if self.data[i] > self.data[i-1] && self.data[i] > self.data[i+1] {
                peaks += 1;
            }
        }
        peaks
    }
    
    /// Calculates average peak width
    fn calculate_average_peak_width(&self) -> f64 {
        // Simplified peak width calculation
        10.0 // Default peak width in data points
    }
    
    /// Counts absorption lines
    fn count_absorption_lines(&self) -> usize {
        let mut lines = 0;
        for i in 1..self.data.len().saturating_sub(1) {
            if self.data[i] < self.data[i-1] && self.data[i] < self.data[i+1] && self.data[i] < 0.3 {
                lines += 1;
            }
        }
        lines
    }
}

// Supporting data structures
#[derive(Debug, Clone, Default)]
pub struct SpectrumMetadata {
    pub measurement_timestamp: Option<chrono::DateTime<chrono::Utc>>,
    pub integration_time_ms: u64,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct SpectralMoments {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl SpectralMoments {
    pub fn weighted_sum(&self) -> f64 {
        self.mean * 0.4 + self.variance * 0.3 + self.skewness.abs() * 0.2 + self.kurtosis.abs() * 0.1
    }
}

#[derive(Debug, Clone, Default)]
pub struct PeakAnalysis {
    pub dominant_peak_intensity: f64,
    pub peak_count: usize,
    pub peak_width_average: f64,
}

#[derive(Debug, Clone, Default)]
pub struct AbsorptionFeatures {
    pub total_absorption: f64,
    pub absorption_lines: usize,
}

#[derive(Debug, Clone)]
pub struct SpaceRegion {
    pub center: Vector3<f64>,
    pub dimensions: Vector3<f64>,
    pub estimated_gas_density: f64,
}

impl SpaceRegion {
    pub fn from_space_id(space_id: SpaceId) -> Self {
        // Convert space ID to spatial region
        // This is a simplified implementation
        let hash = space_id.as_u128() as f64;
        
        Self {
            center: Vector3::new(hash % 10.0, (hash / 10.0) % 10.0, (hash / 100.0) % 10.0),
            dimensions: Vector3::new(1.0, 1.0, 1.0),
            estimated_gas_density: 1.225, // Standard air density
        }
    }
}

#[derive(Debug, Clone)]
pub struct CalibrationData {
    pub calibration_timestamp: chrono::DateTime<chrono::Utc>,
    pub led_calibrations: HashMap<f64, Array1<f64>>,
}

impl CalibrationData {
    pub fn new() -> Self {
        Self {
            calibration_timestamp: chrono::Utc::now(),
            led_calibrations: HashMap::new(),
        }
    }
    
    pub fn add_led_calibration(&mut self, wavelength: f64, spectrum: Array1<f64>) {
        self.led_calibrations.insert(wavelength, spectrum);
    }
}

impl LedSpectrometerReader {
    fn capture_calibration_spectrum(&mut self, controller: &mut LedController) -> Result<Array1<f64>> {
        // Capture calibration spectrum (empty space measurement)
        controller.activate()?;
        std::thread::sleep(Duration::from_millis(100)); // Extended stabilization
        
        let space_region = SpaceRegion {
            center: Vector3::zeros(),
            dimensions: Vector3::new(1.0, 1.0, 1.0),
            estimated_gas_density: 0.0, // Empty space
        };
        
        let spectrum = self.capture_wavelength_response(controller, &space_region)?;
        controller.deactivate()?;
        
        Ok(spectrum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_led_spectrometer_creation() {
        let config = SpectrometryConfig::default();
        let reader = LedSpectrometerReader::new(config);
        assert!(reader.is_ok());
    }
    
    #[test]
    fn test_spectrum_data_analysis() {
        let mut spectrum = SpectrumData::new();
        spectrum.data = vec![0.1, 0.5, 0.8, 0.3, 0.2, 0.9, 0.1];
        
        let moments = spectrum.calculate_moments();
        assert!(moments.mean > 0.0);
        
        let peaks = spectrum.analyze_peaks();
        assert!(peaks.peak_count > 0);
    }
    
    #[test]
    fn test_s_extraction_algorithm() {
        let st_stella = StStellaConstant::golden_ratio();
        let algorithm = SExtractionAlgorithm::new(st_stella).unwrap();
        
        let mut spectrum = SpectrumData::new();
        spectrum.data = vec![0.1, 0.5, 0.8, 0.3, 0.2, 0.9, 0.1];
        
        let s_value = algorithm.extract_s_from_spectrum(spectrum).unwrap();
        assert!(s_value.is_finite());
        assert!(s_value > 0.0);
    }
}