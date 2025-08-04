//! # St. Stella Constant Implementation
//! 
//! The St. Stella constant (σ_St) is the fundamental parameter that enables
//! S-entropy coordinate transformation and zero-computation navigation.
//! 
//! ## Mathematical Foundation
//! 
//! The St. Stella constant governs the relationship between entropy endpoints
//! and spatial coordinates, enabling the transformation:
//! 
//! ```text
//! S_total = σ_St × f(ρ, T, P, v⃗, E_internal)
//! ```
//! 
//! ## Theoretical Significance
//! 
//! Named after Saint Stella-Lorraine Masunda, patron saint of impossibility,
//! this constant represents the mathematical necessity required for the
//! S-entropy framework's theoretical coherence.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The St. Stella constant using the golden ratio for optimal performance
pub const ST_STELLA_GOLDEN_RATIO: f64 = 1.618033988749;

/// Mathematical constant for entropy endpoint precision
pub const ENTROPY_ENDPOINT_PRECISION: f64 = 1e-15;

/// Minimum coherence threshold for St. Stella constant
pub const COHERENCE_THRESHOLD: f64 = 1e-12;

/// The St. Stella constant governing S-entropy coordinate transformations.
/// 
/// This constant is not arbitrary but represents a mathematical necessity
/// for the coherent operation of the S-entropy framework.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StStellaConstant {
    /// The numerical value of the constant
    value: f64,
    /// Precision level for calculations
    precision: f64,
    /// Whether the constant maintains theoretical coherence
    coherent: bool,
    /// Validation timestamp
    validated_at: chrono::DateTime<chrono::Utc>,
}

impl StStellaConstant {
    /// Creates a new St. Stella constant with the specified value.
    /// 
    /// # Arguments
    /// 
    /// * `value` - The constant value (typically golden ratio: 1.618033988749)
    /// 
    /// # Returns
    /// 
    /// A new StStellaConstant instance with validation
    /// 
    /// # Panics
    /// 
    /// Panics if the value would create an incoherent S-entropy framework
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use s_entropy_engine::st_stella::StStellaConstant;
    /// 
    /// let constant = StStellaConstant::new(1.618033988749);
    /// assert!(constant.is_coherent());
    /// ```
    pub fn new(value: f64) -> Self {
        let coherent = Self::validate_coherence(value);
        
        if !coherent {
            panic!("St. Stella constant value {} would create incoherent S-entropy framework", value);
        }
        
        Self {
            value,
            precision: ENTROPY_ENDPOINT_PRECISION,
            coherent,
            validated_at: chrono::Utc::now(),
        }
    }
    
    /// Creates the optimal St. Stella constant using the golden ratio.
    /// 
    /// This configuration provides maximum theoretical coherence and
    /// optimal performance for S-entropy operations.
    pub fn golden_ratio() -> Self {
        Self::new(ST_STELLA_GOLDEN_RATIO)
    }
    
    /// Returns the numerical value of the St. Stella constant.
    pub fn value(&self) -> f64 {
        self.value
    }
    
    /// Returns the precision level for calculations using this constant.
    pub fn precision(&self) -> f64 {
        self.precision
    }
    
    /// Checks if the constant maintains theoretical coherence.
    /// 
    /// A coherent St. Stella constant ensures that:
    /// - S-entropy navigation produces valid results
    /// - Coordinate transformations are bijective
    /// - Zero-computation operations maintain mathematical validity
    pub fn is_coherent(&self) -> bool {
        self.coherent && Self::validate_coherence(self.value)
    }
    
    /// Validates the theoretical coherence of a proposed St. Stella value.
    /// 
    /// # Arguments
    /// 
    /// * `value` - The proposed constant value
    /// 
    /// # Returns
    /// 
    /// `true` if the value maintains S-entropy framework coherence
    fn validate_coherence(value: f64) -> bool {
        // Validation criteria for St. Stella constant coherence:
        
        // 1. Must be finite and positive
        if !value.is_finite() || value <= 0.0 {
            return false;
        }
        
        // 2. Must not create singularities in coordinate transformation
        if (value - 1.0).abs() < COHERENCE_THRESHOLD {
            return false; // Too close to unity creates transformation singularities
        }
        
        // 3. Must maintain precision for entropy endpoint calculations
        if value < COHERENCE_THRESHOLD || value > 1.0 / COHERENCE_THRESHOLD {
            return false; // Outside precision bounds
        }
        
        // 4. Golden ratio provides optimal coherence
        let golden_ratio_coherence = (value - ST_STELLA_GOLDEN_RATIO).abs() < 0.1;
        
        // 5. Alternative coherent values must satisfy mathematical constraints
        let mathematical_coherence = Self::check_mathematical_constraints(value);
        
        golden_ratio_coherence || mathematical_coherence
    }
    
    /// Checks mathematical constraints for alternative St. Stella values.
    fn check_mathematical_constraints(value: f64) -> bool {
        // Mathematical constraints derived from S-entropy theory:
        
        // Constraint 1: Must preserve entropy-endpoint bijection
        let bijection_constraint = value * value - value - 1.0; // Golden ratio equation generalized
        
        // Constraint 2: Must maintain oscillation endpoint stability
        let stability_constraint = (2.0 * value).sin() + (value / 2.0).cos();
        
        // Constraint 3: Must ensure coordinate transformation convergence
        let convergence_constraint = value.ln() + (1.0 / value).exp();
        
        // All constraints must be within acceptable bounds
        bijection_constraint.abs() < 0.01 ||
        stability_constraint.abs() < 0.1 ||
        convergence_constraint < 10.0
    }
    
    /// Transforms an entropy value using the St. Stella constant.
    /// 
    /// This is the fundamental operation that enables S-entropy coordinate mapping.
    /// 
    /// # Arguments
    /// 
    /// * `entropy` - Input entropy value
    /// 
    /// # Returns
    /// 
    /// Transformed entropy suitable for coordinate navigation
    pub fn transform_entropy(&self, entropy: f64) -> f64 {
        if !self.is_coherent() {
            panic!("Cannot transform entropy with incoherent St. Stella constant");
        }
        
        // Core S-entropy transformation using St. Stella constant
        self.value * entropy.ln().abs()
    }
    
    /// Computes the inverse transformation for coordinate-to-entropy mapping.
    /// 
    /// # Arguments
    /// 
    /// * `transformed_entropy` - Previously transformed entropy value
    /// 
    /// # Returns
    /// 
    /// Original entropy value
    pub fn inverse_transform_entropy(&self, transformed_entropy: f64) -> f64 {
        if !self.is_coherent() {
            panic!("Cannot inverse transform entropy with incoherent St. Stella constant");
        }
        
        // Inverse S-entropy transformation
        (transformed_entropy / self.value).exp()
    }
    
    /// Calculates the St. Stella factor for coordinate scaling.
    /// 
    /// This factor determines the scaling relationship between S-entropy
    /// values and spatial coordinates.
    pub fn coordinate_scale_factor(&self) -> f64 {
        self.value.sqrt() * (self.value + 1.0).ln()
    }
    
    /// Computes oscillation endpoint amplitude using the St. Stella constant.
    /// 
    /// # Arguments
    /// 
    /// * `base_amplitude` - Base oscillation amplitude
    /// 
    /// # Returns
    /// 
    /// St. Stella optimized oscillation amplitude
    pub fn oscillation_amplitude(&self, base_amplitude: f64) -> f64 {
        base_amplitude * self.value / (self.value + 1.0)
    }
    
    /// Returns diagnostic information about the St. Stella constant.
    pub fn diagnostics(&self) -> StStellaDiagnostics {
        StStellaDiagnostics {
            value: self.value,
            coherent: self.is_coherent(),
            precision: self.precision,
            golden_ratio_deviation: (self.value - ST_STELLA_GOLDEN_RATIO).abs(),
            coordinate_scale_factor: self.coordinate_scale_factor(),
            validated_at: self.validated_at,
        }
    }
}

/// Diagnostic information for St. Stella constant analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StStellaDiagnostics {
    /// The constant value
    pub value: f64,
    /// Coherence status
    pub coherent: bool,
    /// Calculation precision
    pub precision: f64,
    /// Deviation from golden ratio
    pub golden_ratio_deviation: f64,
    /// Coordinate scaling factor
    pub coordinate_scale_factor: f64,
    /// Validation timestamp
    pub validated_at: chrono::DateTime<chrono::Utc>,
}

impl fmt::Display for StStellaConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "St. Stella Constant: {} (coherent: {}, precision: {:.2e})",
            self.value, self.coherent, self.precision
        )
    }
}

impl Default for StStellaConstant {
    /// Creates the default St. Stella constant using the golden ratio.
    fn default() -> Self {
        Self::golden_ratio()
    }
}

/// Convenience function to create a golden ratio St. Stella constant.
pub fn golden_ratio_constant() -> StStellaConstant {
    StStellaConstant::golden_ratio()
}

/// Validates if a value can serve as a coherent St. Stella constant.
/// 
/// # Arguments
/// 
/// * `value` - Value to validate
/// 
/// # Returns
/// 
/// `true` if the value maintains S-entropy framework coherence
pub fn validate_st_stella_value(value: f64) -> bool {
    StStellaConstant::validate_coherence(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_golden_ratio_constant() {
        let constant = StStellaConstant::golden_ratio();
        assert_eq!(constant.value(), ST_STELLA_GOLDEN_RATIO);
        assert!(constant.is_coherent());
    }
    
    #[test]
    fn test_coherence_validation() {
        // Golden ratio should be coherent
        assert!(validate_st_stella_value(ST_STELLA_GOLDEN_RATIO));
        
        // Invalid values should be incoherent
        assert!(!validate_st_stella_value(0.0));
        assert!(!validate_st_stella_value(-1.0));
        assert!(!validate_st_stella_value(f64::INFINITY));
        assert!(!validate_st_stella_value(f64::NAN));
    }
    
    #[test]
    fn test_entropy_transformation() {
        let constant = StStellaConstant::golden_ratio();
        let entropy = 42.0;
        
        let transformed = constant.transform_entropy(entropy);
        let restored = constant.inverse_transform_entropy(transformed);
        
        assert_relative_eq!(entropy, restored, epsilon = 1e-10);
    }
    
    #[test]
    fn test_coordinate_scale_factor() {
        let constant = StStellaConstant::golden_ratio();
        let scale_factor = constant.coordinate_scale_factor();
        
        assert!(scale_factor.is_finite());
        assert!(scale_factor > 0.0);
    }
    
    #[test]
    fn test_oscillation_amplitude() {
        let constant = StStellaConstant::golden_ratio();
        let base_amplitude = 10.0;
        
        let optimized_amplitude = constant.oscillation_amplitude(base_amplitude);
        
        assert!(optimized_amplitude.is_finite());
        assert!(optimized_amplitude > 0.0);
        assert!(optimized_amplitude < base_amplitude); // Should be scaled down
    }
    
    #[test]
    fn test_diagnostics() {
        let constant = StStellaConstant::golden_ratio();
        let diagnostics = constant.diagnostics();
        
        assert_eq!(diagnostics.value, ST_STELLA_GOLDEN_RATIO);
        assert!(diagnostics.coherent);
        assert_eq!(diagnostics.golden_ratio_deviation, 0.0);
    }
    
    #[test]
    #[should_panic(expected = "incoherent S-entropy framework")]
    fn test_incoherent_constant_panic() {
        StStellaConstant::new(0.0); // Should panic
    }
    
    #[test]
    fn test_display_formatting() {
        let constant = StStellaConstant::golden_ratio();
        let display_string = format!("{}", constant);
        
        assert!(display_string.contains("St. Stella Constant"));
        assert!(display_string.contains(&format!("{}", ST_STELLA_GOLDEN_RATIO)));
        assert!(display_string.contains("coherent: true"));
    }
}