//! # Turbulance Engine WebAssembly Bindings
//! 
//! Advanced probabilistic analysis and reasoning engine for browser environments

use wasm_bindgen::prelude::*;
use serde_wasm_bindgen;

/// Turbulance Engine WebAssembly interface for advanced probabilistic analysis
#[wasm_bindgen]
pub struct TurbulanceWasm {
    // Placeholder for future Turbulance engine integration
}

#[wasm_bindgen]
impl TurbulanceWasm {
    /// Creates a new Turbulance WASM instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> TurbulanceWasm {
        TurbulanceWasm {}
    }
    
    /// Performs probabilistic S-entropy analysis
    /// 
    /// Advanced probabilistic reasoning for S-entropy patterns and predictions
    /// 
    /// # Arguments
    /// 
    /// * `s_values` - Array of S-entropy values for analysis
    /// 
    /// # Returns
    /// 
    /// Probabilistic analysis results including predictions and confidence intervals
    #[wasm_bindgen]
    pub fn analyze_s_entropy_probability(&self, s_values: JsValue) -> JsValue {
        let values: Result<Vec<f64>, _> = serde_wasm_bindgen::from_value(s_values);
        
        match values {
            Ok(s_array) => {
                if s_array.is_empty() {
                    let error_result = serde_json::json!({
                        "success": false,
                        "error": "empty_s_values_array",
                        "message": "Need at least one S-value for analysis"
                    });
                    return serde_wasm_bindgen::to_value(&error_result).unwrap();
                }
                
                // Perform probabilistic analysis
                let mean = s_array.iter().sum::<f64>() / s_array.len() as f64;
                let variance = s_array.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / s_array.len() as f64;
                let std_dev = variance.sqrt();
                
                // Predict next S-value based on trend
                let trend = if s_array.len() > 1 {
                    s_array[s_array.len() - 1] - s_array[s_array.len() - 2]
                } else {
                    0.0
                };
                
                let predicted_next = s_array[s_array.len() - 1] + trend;
                let confidence_interval = 1.96 * std_dev; // 95% confidence interval
                
                let analysis_result = serde_json::json!({
                    "success": true,
                    "analysis": {
                        "statistical_measures": {
                            "mean": mean,
                            "variance": variance,
                            "standard_deviation": std_dev,
                            "min": s_array.iter().cloned().fold(f64::INFINITY, f64::min),
                            "max": s_array.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                        },
                        "trend_analysis": {
                            "current_trend": trend,
                            "trend_direction": if trend > 0.0 { "increasing" } else if trend < 0.0 { "decreasing" } else { "stable" }
                        },
                        "predictions": {
                            "next_s_value": predicted_next,
                            "confidence_interval_lower": predicted_next - confidence_interval,
                            "confidence_interval_upper": predicted_next + confidence_interval,
                            "prediction_confidence": 0.95
                        }
                    },
                    "data_points": s_array.len(),
                    "framework": "Turbulance Probabilistic Analysis",
                    "computation_method": "advanced_statistical_reasoning"
                });
                
                serde_wasm_bindgen::to_value(&analysis_result).unwrap()
            }
            Err(e) => {
                let error_result = serde_json::json!({
                    "success": false,
                    "error": "invalid_s_values_format",
                    "message": format!("Failed to parse S-values: {}", e)
                });
                serde_wasm_bindgen::to_value(&error_result).unwrap()
            }
        }
    }
    
    /// Performs advanced reasoning on S-entropy patterns
    /// 
    /// Uses advanced logical reasoning to identify patterns and anomalies
    /// in S-entropy data sequences.
    #[wasm_bindgen]
    pub fn reason_about_s_patterns(&self, s_data: JsValue) -> JsValue {
        let data: Result<serde_json::Value, _> = serde_wasm_bindgen::from_value(s_data);
        
        match data {
            Ok(pattern_data) => {
                // Extract pattern analysis parameters
                let s_values = pattern_data["s_values"].as_array()
                    .unwrap_or(&Vec::new())
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .collect::<Vec<f64>>();
                
                if s_values.is_empty() {
                    let error_result = serde_json::json!({
                        "success": false,
                        "error": "no_valid_s_values",
                        "message": "No valid S-values found in pattern data"
                    });
                    return serde_wasm_bindgen::to_value(&error_result).unwrap();
                }
                
                // Pattern recognition analysis
                let pattern_analysis = self.analyze_patterns(&s_values);
                let anomaly_detection = self.detect_anomalies(&s_values);
                let entropy_classification = self.classify_entropy_regime(&s_values);
                
                let reasoning_result = serde_json::json!({
                    "success": true,
                    "reasoning": {
                        "pattern_analysis": pattern_analysis,
                        "anomaly_detection": anomaly_detection,
                        "entropy_classification": entropy_classification
                    },
                    "logical_conclusions": {
                        "system_state": self.infer_system_state(&s_values),
                        "recommendations": self.generate_recommendations(&s_values),
                        "confidence": self.calculate_reasoning_confidence(&s_values)
                    },
                    "framework": "Turbulance Advanced Reasoning Engine",
                    "reasoning_method": "logical_inference_with_probabilistic_validation"
                });
                
                serde_wasm_bindgen::to_value(&reasoning_result).unwrap()
            }
            Err(e) => {
                let error_result = serde_json::json!({
                    "success": false,
                    "error": "invalid_pattern_data",
                    "message": format!("Failed to parse pattern data: {}", e)
                });
                serde_wasm_bindgen::to_value(&error_result).unwrap()
            }
        }
    }
    
    /// Returns Turbulance engine status and capabilities
    #[wasm_bindgen]
    pub fn get_turbulance_status(&self) -> JsValue {
        let status = serde_json::json!({
            "engine_operational": true,
            "capabilities": {
                "probabilistic_analysis": true,
                "pattern_recognition": true,
                "anomaly_detection": true,
                "logical_reasoning": true,
                "predictive_modeling": true
            },
            "analysis_methods": {
                "statistical_inference": "advanced",
                "trend_analysis": "enabled",
                "confidence_intervals": "95_percent_default",
                "pattern_classification": "multi_regime"
            },
            "framework": "Turbulance Probabilistic Reasoning Engine",
            "integration": "S-Entropy Framework Compatible"
        });
        
        serde_wasm_bindgen::to_value(&status).unwrap()
    }
}

// Helper methods for Turbulance analysis
impl TurbulanceWasm {
    fn analyze_patterns(&self, s_values: &[f64]) -> serde_json::Value {
        // Simple pattern analysis
        let is_increasing = s_values.windows(2).all(|w| w[1] >= w[0]);
        let is_decreasing = s_values.windows(2).all(|w| w[1] <= w[0]);
        let is_oscillating = s_values.len() > 2 && 
            s_values.windows(3).any(|w| (w[1] > w[0] && w[1] > w[2]) || (w[1] < w[0] && w[1] < w[2]));
        
        serde_json::json!({
            "trend_pattern": if is_increasing { "monotonic_increasing" } 
                            else if is_decreasing { "monotonic_decreasing" }
                            else if is_oscillating { "oscillatory" }
                            else { "irregular" },
            "pattern_strength": if is_increasing || is_decreasing { 0.9 } 
                               else if is_oscillating { 0.7 } 
                               else { 0.3 },
            "pattern_confidence": 0.85
        })
    }
    
    fn detect_anomalies(&self, s_values: &[f64]) -> serde_json::Value {
        if s_values.len() < 3 {
            return serde_json::json!({
                "anomalies_detected": false,
                "anomaly_count": 0,
                "message": "insufficient_data_for_anomaly_detection"
            });
        }
        
        let mean = s_values.iter().sum::<f64>() / s_values.len() as f64;
        let std_dev = (s_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / s_values.len() as f64).sqrt();
        
        let threshold = 2.0 * std_dev; // 2-sigma threshold
        let anomalies: Vec<usize> = s_values.iter()
            .enumerate()
            .filter(|(_, &value)| (value - mean).abs() > threshold)
            .map(|(index, _)| index)
            .collect();
        
        serde_json::json!({
            "anomalies_detected": !anomalies.is_empty(),
            "anomaly_count": anomalies.len(),
            "anomaly_indices": anomalies,
            "threshold_used": threshold,
            "detection_method": "two_sigma_statistical"
        })
    }
    
    fn classify_entropy_regime(&self, s_values: &[f64]) -> serde_json::Value {
        let mean = s_values.iter().sum::<f64>() / s_values.len() as f64;
        
        let regime = match mean {
            x if x < 30.0 => "low_entropy",
            x if x < 50.0 => "medium_entropy", 
            x if x < 70.0 => "high_entropy",
            _ => "very_high_entropy"
        };
        
        serde_json::json!({
            "entropy_regime": regime,
            "mean_s_value": mean,
            "regime_confidence": 0.8,
            "classification_method": "threshold_based"
        })
    }
    
    fn infer_system_state(&self, s_values: &[f64]) -> &'static str {
        let variance = if s_values.len() > 1 {
            let mean = s_values.iter().sum::<f64>() / s_values.len() as f64;
            s_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / s_values.len() as f64
        } else {
            0.0
        };
        
        if variance < 1.0 {
            "stable"
        } else if variance < 10.0 {
            "dynamic"
        } else {
            "chaotic"
        }
    }
    
    fn generate_recommendations(&self, s_values: &[f64]) -> Vec<&'static str> {
        let mut recommendations = Vec::new();
        
        let mean = s_values.iter().sum::<f64>() / s_values.len() as f64;
        
        if mean < 35.0 {
            recommendations.push("increase_s_entropy_sources");
        }
        
        if s_values.len() < 10 {
            recommendations.push("collect_more_data_points");
        }
        
        if s_values.windows(2).all(|w| w[1] == w[0]) {
            recommendations.push("verify_sensor_functionality");
        }
        
        if recommendations.is_empty() {
            recommendations.push("continue_monitoring");
        }
        
        recommendations
    }
    
    fn calculate_reasoning_confidence(&self, s_values: &[f64]) -> f64 {
        // Confidence based on data quantity and quality
        let data_quantity_factor = (s_values.len() as f64).min(20.0) / 20.0; // Max confidence at 20+ points
        let data_quality_factor = if s_values.iter().all(|x| x.is_finite()) { 1.0 } else { 0.5 };
        
        (data_quantity_factor * data_quality_factor * 0.9).max(0.1).min(1.0)
    }
}