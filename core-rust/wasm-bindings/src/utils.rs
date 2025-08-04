//! # WebAssembly Utilities
//! 
//! Common utilities and helper functions for S-Entropy WASM bindings

use wasm_bindgen::prelude::*;
use web_sys::console;

/// Utility functions for S-Entropy WebAssembly operations
#[wasm_bindgen]
pub struct WasmUtils;

#[wasm_bindgen]
impl WasmUtils {
    /// Validates S-entropy value format and range
    /// 
    /// Ensures S-entropy values are mathematically valid for framework operations
    #[wasm_bindgen]
    pub fn validate_s_value(s_value: f64) -> bool {
        s_value.is_finite() && !s_value.is_nan()
    }
    
    /// Converts JavaScript array to validated S-entropy values
    /// 
    /// Filters and validates S-entropy values from JavaScript arrays
    #[wasm_bindgen]
    pub fn parse_s_values_array(js_array: JsValue) -> JsValue {
        match serde_wasm_bindgen::from_value::<Vec<f64>>(js_array) {
            Ok(values) => {
                let valid_values: Vec<f64> = values.into_iter()
                    .filter(|&v| Self::validate_s_value(v))
                    .collect();
                
                let result = serde_json::json!({
                    "success": true,
                    "valid_values": valid_values,
                    "count": valid_values.len()
                });
                
                serde_wasm_bindgen::to_value(&result).unwrap()
            }
            Err(e) => {
                let error_result = serde_json::json!({
                    "success": false,
                    "error": "invalid_array_format",
                    "message": e.to_string()
                });
                
                serde_wasm_bindgen::to_value(&error_result).unwrap()
            }
        }
    }
    
    /// Logs S-entropy framework messages to browser console
    #[wasm_bindgen]
    pub fn log_s_entropy_message(level: &str, message: &str) {
        match level {
            "info" => console::info_1(&format!("[S-Entropy] {}", message).into()),
            "warn" => console::warn_1(&format!("[S-Entropy] {}", message).into()),
            "error" => console::error_1(&format!("[S-Entropy] {}", message).into()),
            _ => console::log_1(&format!("[S-Entropy] {}", message).into()),
        }
    }
    
    /// Formats performance timing for display
    #[wasm_bindgen]
    pub fn format_performance_timing(nanoseconds: f64) -> String {
        if nanoseconds == 0.0 {
            "Zero computation (O(0))".to_string()
        } else if nanoseconds < 1000.0 {
            format!("{:.2} ns", nanoseconds)
        } else if nanoseconds < 1_000_000.0 {
            format!("{:.2} µs", nanoseconds / 1000.0)
        } else if nanoseconds < 1_000_000_000.0 {
            format!("{:.2} ms", nanoseconds / 1_000_000.0)
        } else {
            format!("{:.2} s", nanoseconds / 1_000_000_000.0)
        }
    }
    
    /// Generates universally unique identifier for spatial regions
    #[wasm_bindgen]
    pub fn generate_uuid() -> String {
        uuid::Uuid::new_v4().to_string()
    }
    
    /// Gets current timestamp in RFC3339 format
    #[wasm_bindgen]
    pub fn get_current_timestamp() -> String {
        chrono::Utc::now().to_rfc3339()
    }
    
    /// Calculates memory improvement factor for display
    #[wasm_bindgen]
    pub fn format_memory_improvement(traditional_bytes: f64, s_entropy_bytes: f64) -> String {
        if s_entropy_bytes == 0.0 {
            "Infinite improvement".to_string()
        } else {
            let factor = traditional_bytes / s_entropy_bytes;
            if factor >= 1e12 {
                format!("{:.1}T× improvement", factor / 1e12)
            } else if factor >= 1e9 {
                format!("{:.1}G× improvement", factor / 1e9)
            } else if factor >= 1e6 {
                format!("{:.1}M× improvement", factor / 1e6)
            } else if factor >= 1e3 {
                format!("{:.1}K× improvement", factor / 1e3)
            } else {
                format!("{:.1}× improvement", factor)
            }
        }
    }
    
    /// Creates error response object
    #[wasm_bindgen]
    pub fn create_error_response(error_type: &str, message: &str) -> JsValue {
        let error_response = serde_json::json!({
            "success": false,
            "error": error_type,
            "message": message,
            "timestamp": Self::get_current_timestamp(),
            "framework": "S-Entropy WASM"
        });
        
        serde_wasm_bindgen::to_value(&error_response).unwrap()
    }
    
    /// Creates success response object
    #[wasm_bindgen]
    pub fn create_success_response(data: JsValue) -> JsValue {
        let response = serde_json::json!({
            "success": true,
            "data": serde_wasm_bindgen::from_value::<serde_json::Value>(data).unwrap_or(serde_json::Value::Null),
            "timestamp": Self::get_current_timestamp(),
            "framework": "S-Entropy WASM"
        });
        
        serde_wasm_bindgen::to_value(&response).unwrap()
    }
}

/// Set panic hook for better error reporting in browser
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Initialize WebAssembly utilities
#[wasm_bindgen(start)]
pub fn init_utils() {
    set_panic_hook();
}

/// Browser-specific performance timing utilities
#[wasm_bindgen]
pub fn get_high_resolution_time() -> f64 {
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
}

/// Memory usage estimation for JavaScript objects
#[wasm_bindgen]
pub fn estimate_js_object_size(js_value: JsValue) -> usize {
    // Rough estimation based on JSON serialization
    match serde_wasm_bindgen::from_value::<serde_json::Value>(js_value) {
        Ok(value) => {
            serde_json::to_string(&value)
                .map(|s| s.len())
                .unwrap_or(0)
        }
        Err(_) => 0
    }
}

/// Validates browser compatibility for S-Entropy features
#[wasm_bindgen]
pub fn check_browser_compatibility() -> JsValue {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();
    
    let compatibility = serde_json::json!({
        "webassembly_supported": js_sys::eval("typeof WebAssembly !== 'undefined'")
            .map(|v| v.as_bool().unwrap_or(false))
            .unwrap_or(false),
        "performance_api_available": window.performance().is_ok(),
        "console_api_available": true, // Always available in browser
        "json_support": js_sys::eval("typeof JSON !== 'undefined'")
            .map(|v| v.as_bool().unwrap_or(false))
            .unwrap_or(false),
        "geolocation_supported": navigator.geolocation().is_ok(),
        "media_devices_supported": navigator.media_devices().is_ok(),
        "network_info_supported": navigator.connection().is_some(),
        "web_workers_supported": js_sys::eval("typeof Worker !== 'undefined'")
            .map(|v| v.as_bool().unwrap_or(false))
            .unwrap_or(false),
        "s_entropy_framework_compatible": true
    });
    
    serde_wasm_bindgen::to_value(&compatibility).unwrap()
}

/// Converts coordinates to different formats for JavaScript consumption
#[wasm_bindgen]
pub fn format_coordinates_for_js(x: f64, y: f64, z: f64) -> JsValue {
    let coords = serde_json::json!({
        "cartesian": { "x": x, "y": y, "z": z },
        "spherical": {
            "r": (x*x + y*y + z*z).sqrt(),
            "theta": y.atan2(x),
            "phi": (z / (x*x + y*y + z*z).sqrt()).acos()
        },
        "magnitude": (x*x + y*y + z*z).sqrt()
    });
    
    serde_wasm_bindgen::to_value(&coords).unwrap()
}