//! # Turbulance Type System
//! 
//! Type definitions and value representations for the Turbulance runtime.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Runtime value representation in Turbulance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Null/undefined value
    Null,
    
    /// Boolean value
    Boolean(bool),
    
    /// Integer value
    Integer(i64),
    
    /// Floating point value with optional uncertainty
    Float(f64, Option<Uncertainty>),
    
    /// String value
    String(String),
    
    /// Array/list of values
    Array(Vec<Value>),
    
    /// Object/dictionary of key-value pairs
    Object(HashMap<String, Value>),
    
    /// Function value (closure)
    Function(FunctionValue),
    
    /// Probabilistic value with distribution
    Probabilistic(ProbabilisticValue),
    
    /// Evidence collection
    Evidence(EvidenceValue),
    
    /// Goal with optimization state
    Goal(GoalValue),
    
    /// Proposition with current evidence state
    Proposition(PropositionValue),
    
    /// Motion with support/contradiction evidence
    Motion(MotionValue),
    
    /// Temporal data with time series
    Temporal(TemporalValue),
    
    /// Pattern with matching criteria
    Pattern(PatternValue),
    
    /// Verification result
    Verification(VerificationValue),
    
    /// Reference to another value
    Reference(Uuid),
}

/// Uncertainty representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Uncertainty {
    /// Type of uncertainty model
    pub uncertainty_type: UncertaintyType,
    
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    
    /// Source of uncertainty
    pub source: Option<String>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of uncertainty models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UncertaintyType {
    /// Simple range: value ± range
    Range { range: f64 },
    
    /// Gaussian distribution
    Gaussian { mean: f64, std_dev: f64 },
    
    /// Uniform distribution
    Uniform { min: f64, max: f64 },
    
    /// Beta distribution
    Beta { alpha: f64, beta: f64 },
    
    /// Custom distribution with parameters
    Custom { 
        distribution: String, 
        parameters: HashMap<String, f64> 
    },
    
    /// Confidence interval
    ConfidenceInterval { 
        lower: f64, 
        upper: f64, 
        confidence_level: f64 
    },
    
    /// Empirical distribution from samples
    Empirical { samples: Vec<f64> },
}

/// Function value representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionValue {
    /// Function name
    pub name: String,
    
    /// Parameter names
    pub parameters: Vec<String>,
    
    /// Function body (simplified as expression tree)
    pub body: String, // In practice, this would be an AST
    
    /// Captured variables (closure)
    pub closure: HashMap<String, Value>,
    
    /// Whether function is async
    pub is_async: bool,
    
    /// Uncertainty propagation method
    pub uncertainty_propagation: Option<String>,
}

/// Probabilistic value with distribution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbabilisticValue {
    /// Distribution type
    pub distribution: String,
    
    /// Distribution parameters
    pub parameters: HashMap<String, f64>,
    
    /// Sampled values (for Monte Carlo)
    pub samples: Option<Vec<f64>>,
    
    /// Confidence level
    pub confidence: f64,
    
    /// Generation timestamp
    pub timestamp: DateTime<Utc>,
}

/// Evidence collection value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceValue {
    /// Evidence identifier
    pub id: Uuid,
    
    /// Evidence sources
    pub sources: Vec<EvidenceSource>,
    
    /// Collected data points
    pub data: Vec<EvidenceDataPoint>,
    
    /// Quality metrics
    pub quality: EvidenceQuality,
    
    /// Validation status
    pub validation_status: ValidationStatus,
    
    /// Collection timestamp
    pub collected_at: DateTime<Utc>,
}

/// Evidence source information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSource {
    /// Source name/identifier
    pub name: String,
    
    /// Source type (sensor, human, model, etc.)
    pub source_type: String,
    
    /// Reliability score (0.0 to 1.0)
    pub reliability: f64,
    
    /// Known biases
    pub biases: Vec<String>,
    
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// Individual evidence data point
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceDataPoint {
    /// Data point value
    pub value: Value,
    
    /// Confidence in this data point
    pub confidence: f64,
    
    /// Source of this data point
    pub source: String,
    
    /// Timestamp when collected
    pub timestamp: DateTime<Utc>,
    
    /// Additional context
    pub context: HashMap<String, Value>,
}

/// Evidence quality assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceQuality {
    /// Completeness score (0.0 to 1.0)
    pub completeness: f64,
    
    /// Accuracy score (0.0 to 1.0)
    pub accuracy: f64,
    
    /// Consistency score (0.0 to 1.0)
    pub consistency: f64,
    
    /// Timeliness score (0.0 to 1.0)
    pub timeliness: f64,
    
    /// Overall quality score
    pub overall: f64,
    
    /// Quality issues identified
    pub issues: Vec<String>,
}

/// Validation status for evidence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Pending,
    Valid,
    Invalid { reason: String },
    Questionable { warnings: Vec<String> },
    RequiresReview,
}

/// Goal value with optimization state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalValue {
    /// Goal identifier
    pub id: Uuid,
    
    /// Goal description
    pub description: String,
    
    /// Current progress (0.0 to 1.0)
    pub progress: f64,
    
    /// Success threshold
    pub success_threshold: f64,
    
    /// Objectives and their current values
    pub objectives: HashMap<String, ObjectiveState>,
    
    /// Optimization history
    pub optimization_history: Vec<OptimizationStep>,
    
    /// Current status
    pub status: GoalStatus,
    
    /// Sub-goals
    pub sub_goals: Vec<Uuid>,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

/// State of an individual objective
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveState {
    /// Current value
    pub current_value: f64,
    
    /// Target value
    pub target_value: f64,
    
    /// Progress toward target (0.0 to 1.0)
    pub progress: f64,
    
    /// Confidence in current value
    pub confidence: f64,
    
    /// Weight in overall optimization
    pub weight: f64,
}

/// Single step in optimization history
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationStep {
    /// Step number
    pub step: usize,
    
    /// Objective values at this step
    pub objective_values: HashMap<String, f64>,
    
    /// Overall fitness score
    pub fitness: f64,
    
    /// Changes made in this step
    pub changes: Vec<String>,
    
    /// Timestamp of this step
    pub timestamp: DateTime<Utc>,
}

/// Goal status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GoalStatus {
    Active,
    Paused,
    Completed,
    Failed { reason: String },
    Cancelled,
}

/// Proposition value with evidence state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropositionValue {
    /// Proposition identifier
    pub id: Uuid,
    
    /// Proposition name
    pub name: String,
    
    /// Associated motions
    pub motions: HashMap<String, Uuid>,
    
    /// Overall confidence in proposition
    pub confidence: f64,
    
    /// Evidence supporting the proposition
    pub supporting_evidence: Vec<Uuid>,
    
    /// Evidence contradicting the proposition
    pub contradicting_evidence: Vec<Uuid>,
    
    /// Evaluation history
    pub evaluation_history: Vec<EvaluationResult>,
    
    /// Current status
    pub status: PropositionStatus,
}

/// Motion value with support state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionValue {
    /// Motion identifier
    pub id: Uuid,
    
    /// Motion name
    pub name: String,
    
    /// Motion description
    pub description: String,
    
    /// Current support level (0.0 to 1.0)
    pub support_level: f64,
    
    /// Current contradiction level (0.0 to 1.0)
    pub contradiction_level: f64,
    
    /// Net support (support - contradiction)
    pub net_support: f64,
    
    /// Confidence in support assessment
    pub confidence: f64,
    
    /// Supporting evidence with weights
    pub supporting_evidence: Vec<WeightedEvidence>,
    
    /// Contradicting evidence with weights
    pub contradicting_evidence: Vec<WeightedEvidence>,
}

/// Weighted evidence for motions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedEvidence {
    /// Evidence identifier
    pub evidence_id: Uuid,
    
    /// Weight of this evidence (0.0 to 1.0)
    pub weight: f64,
    
    /// Confidence in this evidence
    pub confidence: f64,
    
    /// Source of the weighting
    pub weight_source: String,
}

/// Evaluation result for propositions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Evaluation timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Overall confidence score
    pub confidence: f64,
    
    /// Motion-level results
    pub motion_results: HashMap<String, f64>,
    
    /// Evidence used in evaluation
    pub evidence_used: Vec<Uuid>,
    
    /// Evaluation method
    pub method: String,
    
    /// Any warnings or issues
    pub warnings: Vec<String>,
}

/// Proposition status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropositionStatus {
    Active,
    Supported { confidence: f64 },
    Contradicted { confidence: f64 },
    Inconclusive,
    RequiresMoreEvidence,
}

/// Temporal value with time series data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalValue {
    /// Time series data points
    pub data_points: Vec<TemporalDataPoint>,
    
    /// Temporal patterns detected
    pub patterns: Vec<TemporalPattern>,
    
    /// Sampling frequency
    pub frequency: Option<f64>,
    
    /// Time range
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    
    /// Interpolation method
    pub interpolation: Option<String>,
}

/// Single data point in time series
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Value at this time
    pub value: Value,
    
    /// Confidence in this measurement
    pub confidence: f64,
    
    /// Source of this measurement
    pub source: Option<String>,
}

/// Temporal pattern
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// Pattern type (periodic, trending, anomalous, etc.)
    pub pattern_type: String,
    
    /// Pattern parameters
    pub parameters: HashMap<String, f64>,
    
    /// Confidence in pattern detection
    pub confidence: f64,
    
    /// Time range where pattern applies
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
}

/// Pattern value for pattern matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternValue {
    /// Pattern identifier
    pub id: Uuid,
    
    /// Pattern expression
    pub expression: String,
    
    /// Pattern type
    pub pattern_type: String,
    
    /// Matching criteria
    pub criteria: HashMap<String, Value>,
    
    /// Match history
    pub matches: Vec<PatternMatch>,
    
    /// Pattern effectiveness metrics
    pub effectiveness: f64,
}

/// Single pattern match result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternMatch {
    /// Match confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Matched data
    pub matched_data: Value,
    
    /// Match timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Context of the match
    pub context: HashMap<String, Value>,
}

/// Verification result value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationValue {
    /// Verification identifier
    pub id: Uuid,
    
    /// Whether AI understanding was verified
    pub understood: bool,
    
    /// Confidence in verification (0.0 to 1.0)
    pub confidence: f64,
    
    /// Similarity score for generated vs actual
    pub similarity_score: f64,
    
    /// Verification method used
    pub method: String,
    
    /// Verification level (basic, standard, comprehensive)
    pub level: String,
    
    /// Time taken for verification
    pub verification_time: std::time::Duration,
    
    /// Any error messages
    pub error_message: Option<String>,
    
    /// Generated artifacts (images, descriptions, etc.)
    pub artifacts: HashMap<String, Value>,
    
    /// Verification timestamp
    pub timestamp: DateTime<Utc>,
}

/// Type information for values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Primary type name
    pub type_name: String,
    
    /// Generic type parameters
    pub generic_params: Vec<TypeInfo>,
    
    /// Whether the type is optional/nullable
    pub optional: bool,
    
    /// Uncertainty constraints
    pub uncertainty_constraints: Option<UncertaintyConstraints>,
    
    /// Value constraints
    pub value_constraints: Option<ValueConstraints>,
}

/// Constraints on uncertainty for types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyConstraints {
    /// Maximum allowed uncertainty
    pub max_uncertainty: Option<f64>,
    
    /// Minimum confidence required
    pub min_confidence: Option<f64>,
    
    /// Allowed uncertainty types
    pub allowed_types: Option<Vec<String>>,
}

/// Constraints on values for types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueConstraints {
    /// Minimum value (for numeric types)
    pub min_value: Option<f64>,
    
    /// Maximum value (for numeric types)
    pub max_value: Option<f64>,
    
    /// Allowed values (for enum-like types)
    pub allowed_values: Option<Vec<Value>>,
    
    /// Regular expression pattern (for strings)
    pub pattern: Option<String>,
    
    /// Required fields (for objects)
    pub required_fields: Option<Vec<String>>,
}

impl Value {
    /// Get the type of this value
    pub fn get_type(&self) -> TypeInfo {
        match self {
            Value::Null => TypeInfo::null(),
            Value::Boolean(_) => TypeInfo::boolean(),
            Value::Integer(_) => TypeInfo::integer(),
            Value::Float(_, uncertainty) => TypeInfo::float(uncertainty.is_some()),
            Value::String(_) => TypeInfo::string(),
            Value::Array(values) => {
                if let Some(first) = values.first() {
                    TypeInfo::array(first.get_type())
                } else {
                    TypeInfo::array(TypeInfo::null())
                }
            }
            Value::Object(_) => TypeInfo::object(),
            Value::Function(_) => TypeInfo::function(),
            Value::Probabilistic(_) => TypeInfo::probabilistic(),
            Value::Evidence(_) => TypeInfo::evidence(),
            Value::Goal(_) => TypeInfo::goal(),
            Value::Proposition(_) => TypeInfo::proposition(),
            Value::Motion(_) => TypeInfo::motion(),
            Value::Temporal(_) => TypeInfo::temporal(),
            Value::Pattern(_) => TypeInfo::pattern(),
            Value::Verification(_) => TypeInfo::verification(),
            Value::Reference(_) => TypeInfo::reference(),
        }
    }
    
    /// Check if this value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Boolean(b) => *b,
            Value::Integer(i) => *i != 0,
            Value::Float(f, _) => *f != 0.0 && !f.is_nan(),
            Value::String(s) => !s.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            Value::Object(obj) => !obj.is_empty(),
            _ => true, // Complex types are generally truthy if they exist
        }
    }
    
    /// Get confidence level if this value has uncertainty
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Value::Float(_, Some(uncertainty)) => Some(uncertainty.confidence),
            Value::Probabilistic(prob) => Some(prob.confidence),
            Value::Evidence(evidence) => Some(evidence.quality.overall),
            Value::Goal(goal) => Some(goal.progress),
            Value::Proposition(prop) => Some(prop.confidence),
            Value::Motion(motion) => Some(motion.confidence),
            Value::Verification(verif) => Some(verif.confidence),
            _ => None,
        }
    }
    
    /// Convert to string representation
    pub fn to_string_representation(&self) -> String {
        match self {
            Value::Null => "null".to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f, uncertainty) => {
                if let Some(unc) = uncertainty {
                    format!("{} ± {} (conf: {:.2})", f, 
                           Self::uncertainty_to_string(&unc.uncertainty_type), 
                           unc.confidence)
                } else {
                    f.to_string()
                }
            }
            Value::String(s) => format!("\"{}\"", s),
            Value::Array(arr) => {
                let elements: Vec<String> = arr.iter().map(|v| v.to_string_representation()).collect();
                format!("[{}]", elements.join(", "))
            }
            Value::Object(obj) => {
                let pairs: Vec<String> = obj.iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_string_representation()))
                    .collect();
                format!("{{{}}}", pairs.join(", "))
            }
            Value::Function(func) => format!("funxn {}({})", func.name, func.parameters.join(", ")),
            Value::Probabilistic(prob) => format!("~{}({:?})", prob.distribution, prob.parameters),
            Value::Evidence(evidence) => format!("Evidence({})", evidence.id),
            Value::Goal(goal) => format!("Goal({}, {:.1}% complete)", goal.description, goal.progress * 100.0),
            Value::Proposition(prop) => format!("Proposition({}, conf: {:.2})", prop.name, prop.confidence),
            Value::Motion(motion) => format!("Motion({}, support: {:.2})", motion.name, motion.support_level),
            Value::Temporal(temporal) => format!("TimeSeries({} points)", temporal.data_points.len()),
            Value::Pattern(pattern) => format!("Pattern({})", pattern.expression),
            Value::Verification(verif) => format!("Verification({}%, {})", (verif.confidence * 100.0) as i32, verif.method),
            Value::Reference(uuid) => format!("&{}", uuid),
        }
    }
    
    fn uncertainty_to_string(uncertainty_type: &UncertaintyType) -> String {
        match uncertainty_type {
            UncertaintyType::Range { range } => range.to_string(),
            UncertaintyType::Gaussian { std_dev, .. } => format!("σ{}", std_dev),
            UncertaintyType::ConfidenceInterval { lower, upper, .. } => format!("[{}, {}]", lower, upper),
            _ => "?".to_string(),
        }
    }
}

impl TypeInfo {
    pub fn null() -> Self {
        Self {
            type_name: "null".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn boolean() -> Self {
        Self {
            type_name: "boolean".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn integer() -> Self {
        Self {
            type_name: "integer".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn float(has_uncertainty: bool) -> Self {
        Self {
            type_name: if has_uncertainty { "uncertain_float" } else { "float" }.to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn string() -> Self {
        Self {
            type_name: "string".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn array(element_type: TypeInfo) -> Self {
        Self {
            type_name: "array".to_string(),
            generic_params: vec![element_type],
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn object() -> Self {
        Self {
            type_name: "object".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn function() -> Self {
        Self {
            type_name: "function".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn probabilistic() -> Self {
        Self {
            type_name: "probabilistic".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn evidence() -> Self {
        Self {
            type_name: "evidence".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn goal() -> Self {
        Self {
            type_name: "goal".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn proposition() -> Self {
        Self {
            type_name: "proposition".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn motion() -> Self {
        Self {
            type_name: "motion".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn temporal() -> Self {
        Self {
            type_name: "temporal".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn pattern() -> Self {
        Self {
            type_name: "pattern".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn verification() -> Self {
        Self {
            type_name: "verification".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
    
    pub fn reference() -> Self {
        Self {
            type_name: "reference".to_string(),
            generic_params: Vec::new(),
            optional: false,
            uncertainty_constraints: None,
            value_constraints: None,
        }
    }
}

/// Compiled state for propositions, goals, evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropositionState {
    pub proposition: PropositionValue,
    pub execution_context: HashMap<String, Value>,
    pub last_evaluation: Option<DateTime<Utc>>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalState {
    pub goal: GoalValue,
    pub optimization_state: OptimizationState,
    pub last_update: Option<DateTime<Utc>>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationState {
    pub current_parameters: HashMap<String, f64>,
    pub gradient: Option<HashMap<String, f64>>,
    pub learning_rate: f64,
    pub convergence_metrics: ConvergenceMetrics,
    pub iteration_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    pub fitness_improvement: f64,
    pub parameter_change: f64,
    pub gradient_norm: f64,
    pub converged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceCollection {
    pub evidence: EvidenceValue,
    pub collection_state: CollectionState,
    pub processing_pipeline: Vec<String>,
    pub last_update: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionState {
    Collecting,
    Processing,
    Validated,
    Rejected { reason: String },
    Expired,
}

/// Result types for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropositionResult {
    pub proposition_id: Uuid,
    pub confidence: f64,
    pub motion_results: HashMap<String, MotionResult>,
    pub evidence_count: usize,
    pub evaluation_time: std::time::Duration,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionResult {
    pub motion_id: Uuid,
    pub support_level: f64,
    pub contradiction_level: f64,
    pub net_support: f64,
    pub confidence: f64,
    pub evidence_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalResult {
    pub goal_id: Uuid,
    pub progress: f64,
    pub objective_progress: HashMap<String, f64>,
    pub optimization_steps: usize,
    pub convergence_status: String,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceResult {
    pub evidence_id: Uuid,
    pub quality_score: f64,
    pub validation_status: ValidationStatus,
    pub data_points_count: usize,
    pub processing_time: std::time::Duration,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub verification_id: Uuid,
    pub understood: bool,
    pub confidence: f64,
    pub similarity_score: f64,
    pub method: String,
    pub level: String,
    pub verification_time: std::time::Duration,
    pub error_message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type_detection() {
        let int_val = Value::Integer(42);
        assert_eq!(int_val.get_type().type_name, "integer");
        
        let float_val = Value::Float(3.14, None);
        assert_eq!(float_val.get_type().type_name, "float");
        
        let uncertain_float = Value::Float(3.14, Some(Uncertainty {
            uncertainty_type: UncertaintyType::Range { range: 0.01 },
            confidence: 0.95,
            source: None,
            metadata: HashMap::new(),
        }));
        assert_eq!(uncertain_float.get_type().type_name, "uncertain_float");
    }

    #[test]
    fn test_value_truthiness() {
        assert!(!Value::Null.is_truthy());
        assert!(!Value::Boolean(false).is_truthy());
        assert!(Value::Boolean(true).is_truthy());
        assert!(!Value::Integer(0).is_truthy());
        assert!(Value::Integer(42).is_truthy());
        assert!(Value::String("hello".to_string()).is_truthy());
        assert!(!Value::String("".to_string()).is_truthy());
    }

    #[test]
    fn test_value_confidence() {
        let uncertain_val = Value::Float(1.0, Some(Uncertainty {
            uncertainty_type: UncertaintyType::Range { range: 0.1 },
            confidence: 0.8,
            source: None,
            metadata: HashMap::new(),
        }));
        
        assert_eq!(uncertain_val.confidence(), Some(0.8));
        
        let certain_val = Value::Integer(42);
        assert_eq!(certain_val.confidence(), None);
    }

    #[test]
    fn test_string_representation() {
        let int_val = Value::Integer(42);
        assert_eq!(int_val.to_string_representation(), "42");
        
        let str_val = Value::String("hello".to_string());
        assert_eq!(str_val.to_string_representation(), "\"hello\"");
        
        let array_val = Value::Array(vec![Value::Integer(1), Value::Integer(2)]);
        assert_eq!(array_val.to_string_representation(), "[1, 2]");
    }
} 