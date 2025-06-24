//! # Turbulance Abstract Syntax Tree (AST)
//! 
//! Defines the complete AST structure for the Turbulance probabilistic scientific programming language.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::lexer::TokenPosition;

/// Root of the AST - represents a complete Turbulance program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub items: Vec<Item>,
    pub source_info: Option<SourceInfo>,
}

/// Source information for debugging and error reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub file_path: Option<String>,
    pub source_length: usize,
    pub line_count: usize,
}

/// Top-level items in a Turbulance program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Item {
    /// Variable declaration: `item x = value`
    Variable(VariableDeclaration),
    
    /// Function definition: `funxn name(params): body`
    Function(FunctionDeclaration),
    
    /// Proposition definition
    Proposition(PropositionDeclaration),
    
    /// Goal definition
    Goal(GoalDeclaration),
    
    /// Evidence collection definition
    Evidence(EvidenceDeclaration),
    
    /// Metacognitive analysis definition
    Metacognitive(MetacognitiveDeclaration),
    
    /// Configuration block
    Config(ConfigDeclaration),
    
    /// Data sources definition
    Datasources(DatasourcesDeclaration),
    
    /// Evidence integrator definition
    EvidenceIntegrator(EvidenceIntegratorDeclaration),
    
    /// Orchestrator definition
    Orchestrator(OrchestratorDeclaration),
    
    /// Verification system definition
    VerificationSystem(VerificationSystemDeclaration),
    
    /// Real-time orchestrator definition
    RealTimeOrchestrator(RealTimeOrchestratorDeclaration),
    
    /// Interface definition
    Interface(InterfaceDeclaration),
    
    /// Temporal analysis definition
    Temporal(TemporalDeclaration),
    
    /// Cross-domain analysis definition
    CrossDomainAnalysis(CrossDomainDeclaration),
    
    /// Pattern registry definition
    PatternRegistry(PatternRegistryDeclaration),
    
    /// Import statement
    Import(ImportDeclaration),
    
    /// Expression statement
    Expression(Expression),
}

/// Variable declaration with optional type annotation and uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDeclaration {
    pub name: Identifier,
    pub type_annotation: Option<TypeAnnotation>,
    pub value: Expression,
    pub uncertainty: Option<UncertaintySpec>,
    pub position: Option<TokenPosition>,
}

/// Function declaration with probabilistic parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<TypeAnnotation>,
    pub body: Vec<Statement>,
    pub is_async: bool,
    pub uncertainty_propagation: Option<UncertaintyPropagation>,
    pub position: Option<TokenPosition>,
}

/// Function parameter with optional confidence specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: Identifier,
    pub type_annotation: Option<TypeAnnotation>,
    pub default_value: Option<Expression>,
    pub confidence_requirement: Option<f64>,
}

/// Proposition declaration - core scientific reasoning construct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropositionDeclaration {
    pub name: Identifier,
    pub extends: Option<Identifier>,
    pub context: Vec<ContextDeclaration>,
    pub motions: Vec<MotionDeclaration>,
    pub evidence_evaluation: Vec<EvidenceEvaluation>,
    pub confidence_threshold: Option<f64>,
    pub position: Option<TokenPosition>,
}

/// Motion declaration - sub-hypothesis within a proposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionDeclaration {
    pub name: Identifier,
    pub description: String,
    pub requirements: Option<Vec<Requirement>>,
    pub criteria: Option<Vec<Criterion>>,
    pub patterns: Option<Vec<String>>,
    pub confidence_threshold: Option<f64>,
    pub position: Option<TokenPosition>,
}

/// Context declaration for propositions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextDeclaration {
    pub name: Identifier,
    pub value: Expression,
    pub uncertainty: Option<UncertaintySpec>,
}

/// Evidence evaluation within propositions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEvaluation {
    pub scope: WithinClause,
    pub conditions: Vec<GivenClause>,
    pub position: Option<TokenPosition>,
}

/// Within clause for evidence scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinClause {
    pub scope_expression: Expression,
    pub alias: Option<Identifier>,
}

/// Given clause for conditional evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GivenClause {
    pub condition: Expression,
    pub confidence: Option<ConfidenceSpec>,
    pub actions: Vec<EvidenceAction>,
}

/// Actions taken when evidence conditions are met
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceAction {
    Support {
        motion: Identifier,
        weight: Option<Expression>,
        confidence: Option<ConfidenceSpec>,
    },
    Contradict {
        motion: Identifier,
        weight: Option<Expression>,
        confidence: Option<ConfidenceSpec>,
    },
    UpdateEvidence {
        evidence_type: String,
        value: Expression,
    },
    ExecuteAnalysis {
        analysis: Expression,
    },
}

/// Goal declaration with optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalDeclaration {
    pub name: Identifier,
    pub description: String,
    pub objectives: Vec<Objective>,
    pub success_thresholds: HashMap<String, ThresholdSpec>,
    pub optimization_algorithm: Option<OptimizationSpec>,
    pub personalization_factors: Option<HashMap<String, Expression>>,
    pub adaptation_strategy: Option<AdaptationStrategy>,
    pub sub_goals: Vec<GoalDeclaration>,
    pub position: Option<TokenPosition>,
}

/// Optimization objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    pub name: String,
    pub objective_type: ObjectiveType,
    pub expression: Expression,
    pub weight: Option<f64>,
    pub uncertainty_constraint: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectiveType {
    Maximize,
    Minimize,
    Target(f64),
    Constraint,
}

/// Threshold specification with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdSpec {
    pub value: f64,
    pub uncertainty: Option<f64>,
    pub confidence_level: Option<f64>,
}

/// Optimization algorithm specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSpec {
    pub method: String,
    pub parameters: HashMap<String, Expression>,
    pub convergence_criteria: Option<ConvergenceCriteria>,
}

/// Convergence criteria for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub plateau_detection: Option<PlateauDetection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauDetection {
    pub window_size: usize,
    pub improvement_threshold: f64,
}

/// Adaptation strategy for dynamic goal adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStrategy {
    pub progress_monitoring: Vec<String>,
    pub threshold_adjustment: Option<String>,
    pub goal_refinement: Option<String>,
    pub intervention_triggers: Vec<String>,
}

/// Evidence collection declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceDeclaration {
    pub name: Identifier,
    pub sources: Vec<EvidenceSource>,
    pub collection_strategy: Option<CollectionStrategy>,
    pub processing_pipeline: Option<Vec<ProcessingStep>>,
    pub validation_rules: Option<Vec<ValidationRule>>,
    pub quality_metrics: Option<QualityMetrics>,
    pub position: Option<TokenPosition>,
}

/// Evidence source specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSource {
    pub name: String,
    pub source_type: String,
    pub reliability: Option<f64>,
    pub systematic_bias: Option<String>,
    pub uncertainty_model: Option<String>,
    pub metadata: HashMap<String, Expression>,
}

/// Collection strategy for evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStrategy {
    pub frequency: Option<String>,
    pub duration: Option<String>,
    pub validation_method: Option<String>,
    pub quality_threshold: Option<f64>,
}

/// Processing step in evidence pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStep {
    pub name: String,
    pub operation: String,
    pub parameters: HashMap<String, Expression>,
    pub quality_check: Option<Expression>,
}

/// Validation rule for evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub condition: Expression,
    pub severity: ValidationSeverity,
    pub action: ValidationAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationAction {
    Reject,
    Flag,
    Correct(Expression),
    Reprocess,
}

/// Quality metrics for evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub completeness: Option<f64>,
    pub accuracy: Option<f64>,
    pub consistency: Option<f64>,
    pub timeliness: Option<f64>,
    pub bias_indicators: Vec<String>,
}

/// Metacognitive analysis declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveDeclaration {
    pub name: Identifier,
    pub tracking_dimensions: Vec<TrackingDimension>,
    pub evaluation_methods: Vec<EvaluationMethod>,
    pub adaptation_rules: Vec<AdaptationRule>,
    pub quality_assurance: Option<QualityAssurance>,
    pub position: Option<TokenPosition>,
}

/// Dimension to track in metacognitive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingDimension {
    pub name: String,
    pub metrics: HashMap<String, Expression>,
    pub update_frequency: Option<String>,
}

/// Method for evaluating analysis quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMethod {
    pub name: String,
    pub evaluation_expression: Expression,
    pub threshold: Option<f64>,
    pub dependencies: Vec<String>,
}

/// Rule for adapting analysis based on metacognitive insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRule {
    pub condition: Expression,
    pub actions: Vec<AdaptationAction>,
    pub priority: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationAction {
    AdjustThreshold { parameter: String, adjustment: f64 },
    IncreaseEvidence { source: String, amount: f64 },
    RefineMethodology { method: String, parameters: HashMap<String, Expression> },
    SeekExpertReview { urgency: String },
    ReprocessData { stage: String },
}

/// Quality assurance specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssurance {
    pub minimum_confidence: f64,
    pub bias_detection: Vec<String>,
    pub consistency_checks: Vec<String>,
    pub reproducibility_requirements: Vec<String>,
}

/// Configuration declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigDeclaration {
    pub name: Identifier,
    pub settings: HashMap<String, Expression>,
    pub position: Option<TokenPosition>,
}

/// Data sources declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasourcesDeclaration {
    pub name: Identifier,
    pub sources: HashMap<String, DataSourceSpec>,
    pub position: Option<TokenPosition>,
}

/// Specification for a data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceSpec {
    pub source_path: String,
    pub format: Option<String>,
    pub metadata: HashMap<String, Expression>,
    pub quality_indicators: Option<HashMap<String, f64>>,
}

/// Evidence integrator declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceIntegratorDeclaration {
    pub name: Identifier,
    pub sources: Vec<EvidenceSource>,
    pub fusion_methods: Vec<FusionMethod>,
    pub validation_pipeline: Vec<ValidationStep>,
    pub position: Option<TokenPosition>,
}

/// Method for fusing evidence from multiple sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMethod {
    pub name: String,
    pub method_type: String,
    pub parameters: HashMap<String, Expression>,
    pub uncertainty_handling: Option<UncertaintyHandling>,
}

/// Handling of uncertainty in evidence fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyHandling {
    pub propagation_method: String,
    pub correlation_modeling: Option<String>,
    pub confidence_calibration: Option<String>,
}

/// Validation step in evidence integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStep {
    pub name: String,
    pub validation_type: String,
    pub criteria: Expression,
    pub action_on_failure: ValidationAction,
}

/// Orchestrator declaration for coordinating analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorDeclaration {
    pub name: Identifier,
    pub initialization_steps: Vec<InitializationStep>,
    pub execution_phases: Vec<ExecutionPhase>,
    pub monitoring_strategy: Option<MonitoringStrategy>,
    pub position: Option<TokenPosition>,
}

/// Initialization step for orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializationStep {
    pub name: String,
    pub operation: Expression,
    pub dependencies: Vec<String>,
    pub timeout: Option<std::time::Duration>,
}

/// Execution phase in orchestrated analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPhase {
    pub name: String,
    pub operations: Vec<Operation>,
    pub parallel_execution: bool,
    pub success_criteria: Option<Expression>,
    pub failure_handling: Option<FailureHandling>,
}

/// Operation within an execution phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub name: String,
    pub operation_type: OperationType,
    pub parameters: HashMap<String, Expression>,
    pub expected_duration: Option<std::time::Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    DataProcessing,
    Analysis,
    Validation,
    Optimization,
    Reporting,
    Custom(String),
}

/// Failure handling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureHandling {
    pub retry_policy: Option<RetryPolicy>,
    pub fallback_strategy: Option<String>,
    pub notification_strategy: Option<String>,
}

/// Retry policy for failed operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub backoff_strategy: BackoffStrategy,
    pub retry_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Fixed(std::time::Duration),
    Exponential { initial: std::time::Duration, multiplier: f64 },
    Linear { initial: std::time::Duration, increment: std::time::Duration },
}

/// Monitoring strategy for orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStrategy {
    pub metrics: Vec<String>,
    pub thresholds: HashMap<String, f64>,
    pub alert_conditions: Vec<Expression>,
    pub reporting_frequency: Option<String>,
}

/// Other declaration types (abbreviated for space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSystemDeclaration {
    pub name: Identifier,
    pub verification_methods: Vec<VerificationMethod>,
    pub verification_levels: Vec<VerificationLevel>,
    pub position: Option<TokenPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMethod {
    pub name: String,
    pub method_type: String,
    pub parameters: HashMap<String, Expression>,
    pub confidence_calibration: Option<Expression>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationLevel {
    pub name: String,
    pub requirements: Vec<String>,
    pub validation_time_limit: Option<std::time::Duration>,
    pub use_case: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeOrchestratorDeclaration {
    pub name: Identifier,
    pub stream_processing: HashMap<String, StreamSpec>,
    pub continuous_evaluation: Vec<ContinuousEvaluation>,
    pub predictive_modeling: Option<PredictiveModeling>,
    pub position: Option<TokenPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSpec {
    pub source: String,
    pub processing_function: Expression,
    pub latency_requirement: Option<std::time::Duration>,
    pub frequency: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousEvaluation {
    pub frequency: std::time::Duration,
    pub operations: Vec<Expression>,
    pub trigger_conditions: Vec<Expression>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveModeling {
    pub prediction_horizon: std::time::Duration,
    pub model_type: String,
    pub parameters: HashMap<String, Expression>,
    pub uncertainty_quantification: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceDeclaration {
    pub name: Identifier,
    pub components: Vec<InterfaceComponent>,
    pub interactions: Vec<Interaction>,
    pub position: Option<TokenPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceComponent {
    pub name: String,
    pub component_type: String,
    pub properties: HashMap<String, Expression>,
    pub event_handlers: Vec<EventHandler>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventHandler {
    pub event: String,
    pub handler: Expression,
    pub debounce: Option<std::time::Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub name: String,
    pub interaction_type: String,
    pub parameters: HashMap<String, Expression>,
    pub real_time_update: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDeclaration {
    pub name: Identifier,
    pub scope: TemporalScope,
    pub patterns: Vec<TemporalPattern>,
    pub operations: Vec<TemporalOperation>,
    pub position: Option<TokenPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalScope {
    pub start_time: Expression,
    pub end_time: Expression,
    pub resolution: Expression,
    pub time_zone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub name: String,
    pub pattern_type: String,
    pub parameters: HashMap<String, Expression>,
    pub detection_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalOperation {
    pub name: String,
    pub operation_type: String,
    pub parameters: HashMap<String, Expression>,
    pub time_constraints: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainDeclaration {
    pub name: Identifier,
    pub domain_mappings: Vec<DomainMapping>,
    pub patterns: Vec<CrossDomainPattern>,
    pub integration_rules: Vec<IntegrationRule>,
    pub position: Option<TokenPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainMapping {
    pub from_domain: String,
    pub to_domain: String,
    pub mapping_function: Expression,
    pub validation_method: Option<String>,
    pub confidence_assessment: Option<Expression>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainPattern {
    pub name: String,
    pub pattern_type: String,
    pub domains: Vec<String>,
    pub similarity_metric: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationRule {
    pub name: String,
    pub condition: Expression,
    pub action: Expression,
    pub confidence_requirement: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRegistryDeclaration {
    pub name: Identifier,
    pub categories: HashMap<String, PatternCategory>,
    pub matching_rules: MatchingRules,
    pub relationships: Vec<PatternRelationship>,
    pub position: Option<TokenPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCategory {
    pub patterns: HashMap<String, PatternSpec>,
    pub metadata: HashMap<String, Expression>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSpec {
    pub pattern_type: String,
    pub parameters: HashMap<String, Expression>,
    pub confidence_threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingRules {
    pub threshold: f64,
    pub context_window: Option<usize>,
    pub overlap_policy: String,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRelationship {
    pub from_pattern: String,
    pub to_pattern: String,
    pub relationship_type: String,
    pub strength: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportDeclaration {
    pub module_path: String,
    pub imported_items: Option<Vec<String>>,
    pub alias: Option<String>,
    pub conditional: Option<Expression>,
    pub position: Option<TokenPosition>,
}

/// Expressions in Turbulance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    /// Literal values
    Literal(Literal),
    
    /// Identifier reference
    Identifier(Identifier),
    
    /// Binary operations
    BinaryOp {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },
    
    /// Unary operations
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    
    /// Function call with uncertainty propagation
    FunctionCall {
        function: Box<Expression>,
        arguments: Vec<Expression>,
        uncertainty_propagation: Option<UncertaintyPropagation>,
    },
    
    /// Method call
    MethodCall {
        object: Box<Expression>,
        method: String,
        arguments: Vec<Expression>,
    },
    
    /// Array/list literal
    Array(Vec<Expression>),
    
    /// Object/dictionary literal
    Object(HashMap<String, Expression>),
    
    /// Index access
    Index {
        object: Box<Expression>,
        index: Box<Expression>,
    },
    
    /// Field access
    Field {
        object: Box<Expression>,
        field: String,
    },
    
    /// Conditional expression
    Conditional {
        condition: Box<Expression>,
        then_expr: Box<Expression>,
        else_expr: Option<Box<Expression>>,
    },
    
    /// Lambda function
    Lambda {
        parameters: Vec<Parameter>,
        body: Box<Expression>,
    },
    
    /// Range expression
    Range {
        start: Box<Expression>,
        end: Box<Expression>,
        inclusive: bool,
    },
    
    /// Uncertainty specification
    WithUncertainty {
        value: Box<Expression>,
        uncertainty: UncertaintySpec,
    },
    
    /// Confidence specification
    WithConfidence {
        value: Box<Expression>,
        confidence: ConfidenceSpec,
    },
    
    /// Probabilistic expression
    Probabilistic {
        distribution: String,
        parameters: HashMap<String, Expression>,
    },
    
    /// Pattern matching expression
    PatternMatch {
        value: Box<Expression>,
        pattern: String,
        confidence_threshold: Option<f64>,
    },
}

/// Statements in Turbulance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Statement {
    /// Expression statement
    Expression(Expression),
    
    /// Variable assignment
    Assignment {
        target: Expression,
        value: Expression,
    },
    
    /// Return statement
    Return(Option<Expression>),
    
    /// Break statement
    Break,
    
    /// Continue statement
    Continue,
    
    /// If statement
    If {
        condition: Expression,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
    },
    
    /// While loop
    While {
        condition: Expression,
        body: Vec<Statement>,
    },
    
    /// For loop
    For {
        variable: Identifier,
        iterable: Expression,
        body: Vec<Statement>,
    },
    
    /// Try-catch-finally
    TryCatch {
        try_block: Vec<Statement>,
        catch_blocks: Vec<CatchBlock>,
        finally_block: Option<Vec<Statement>>,
    },
    
    /// Parallel execution block
    Parallel {
        operations: Vec<ParallelOperation>,
        synchronization: Option<String>,
    },
    
    /// Stream processing statement
    Stream {
        stream_spec: StreamSpec,
        processing: Vec<Statement>,
    },
    
    /// Segment analysis
    Segment {
        name: String,
        extraction: Expression,
        analysis: Vec<Statement>,
    },
    
    /// Phase execution
    Phase {
        name: String,
        operations: Vec<Statement>,
        success_criteria: Option<Expression>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatchBlock {
    pub exception_type: Option<String>,
    pub variable: Option<Identifier>,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelOperation {
    pub name: String,
    pub operation: Expression,
    pub dependencies: Vec<String>,
}

/// Literal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
}

/// Identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identifier {
    pub name: String,
    pub position: Option<TokenPosition>,
}

/// Binary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Subtract, Multiply, Divide, Modulo, Power,
    
    // Comparison
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    
    // Logical
    And, Or,
    
    // Probabilistic
    PlusMinus, // For uncertainty ranges like ±0.02
    
    // Assignment
    Assign, AddAssign, SubAssign, MulAssign, DivAssign,
}

/// Unary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not, Minus, Plus,
}

/// Type annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeAnnotation {
    pub type_name: String,
    pub generic_params: Option<Vec<TypeAnnotation>>,
    pub optional: bool,
}

/// Uncertainty specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintySpec {
    /// Simple range: ±value
    Range(f64),
    
    /// Gaussian distribution: mean ± std_dev
    Gaussian { mean: f64, std_dev: f64 },
    
    /// Custom distribution
    Distribution {
        distribution_type: String,
        parameters: HashMap<String, f64>,
    },
    
    /// Confidence interval
    ConfidenceInterval {
        lower: f64,
        upper: f64,
        confidence_level: f64,
    },
}

/// Confidence specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceSpec {
    /// Simple confidence value
    Value(f64),
    
    /// Confidence with source attribution
    WithSource {
        confidence: f64,
        source: String,
    },
    
    /// Computed confidence
    Computed(Expression),
}

/// Uncertainty propagation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyPropagation {
    /// Linear propagation
    Linear,
    
    /// Monte Carlo simulation
    MonteCarlo {
        samples: usize,
        seed: Option<u64>,
    },
    
    /// Polynomial chaos expansion
    PolynomialChaos {
        order: usize,
        quadrature_points: usize,
    },
    
    /// Custom propagation method
    Custom {
        method: String,
        parameters: HashMap<String, Expression>,
    },
}

/// Requirements for motions and goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Requirement {
    pub name: String,
    pub requirement_type: String,
    pub specification: Expression,
    pub mandatory: bool,
}

/// Criteria for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Criterion {
    pub name: String,
    pub expression: Expression,
    pub weight: Option<f64>,
    pub threshold: Option<f64>,
}

impl Program {
    /// Create a new empty program
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            source_info: None,
        }
    }
    
    /// Add an item to the program
    pub fn add_item(&mut self, item: Item) {
        self.items.push(item);
    }
    
    /// Get all propositions in the program
    pub fn propositions(&self) -> Vec<&PropositionDeclaration> {
        self.items.iter().filter_map(|item| {
            if let Item::Proposition(prop) = item {
                Some(prop)
            } else {
                None
            }
        }).collect()
    }
    
    /// Get all goals in the program
    pub fn goals(&self) -> Vec<&GoalDeclaration> {
        self.items.iter().filter_map(|item| {
            if let Item::Goal(goal) = item {
                Some(goal)
            } else {
                None
            }
        }).collect()
    }
    
    /// Get all evidence declarations in the program
    pub fn evidence(&self) -> Vec<&EvidenceDeclaration> {
        self.items.iter().filter_map(|item| {
            if let Item::Evidence(evidence) = item {
                Some(evidence)
            } else {
                None
            }
        }).collect()
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Identifier {
    /// Create a new identifier
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            position: None,
        }
    }
    
    /// Create an identifier with position information
    pub fn with_position(name: impl Into<String>, position: TokenPosition) -> Self {
        Self {
            name: name.into(),
            position: Some(position),
        }
    }
}

impl From<&str> for Identifier {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

impl From<String> for Identifier {
    fn from(name: String) -> Self {
        Self::new(name)
    }
} 