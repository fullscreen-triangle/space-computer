//! # Turbulance Parser
//! 
//! A comprehensive parser and compiler for the Turbulance probabilistic scientific programming language.
//! Converts Turbulance syntax into executable tasks with uncertainty quantification and evidence-based reasoning.

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod compiler;
pub mod executor;
pub mod error;
pub mod types;
pub mod probabilistic;
pub mod verification;
pub mod goals;
pub mod evidence;
pub mod metacognitive;

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub use ast::*;
pub use lexer::*;
pub use parser::*;
pub use compiler::*;
pub use executor::*;
pub use error::*;
pub use types::*;

/// Main entry point for parsing Turbulance code
#[derive(Debug, Clone)]
pub struct TurbulanceEngine {
    pub config: EngineConfig,
    pub context: ExecutionContext,
}

/// Configuration for the Turbulance engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub uncertainty_model: String,
    pub confidence_threshold: f64,
    pub verification_required: bool,
    pub real_time_analysis: bool,
    pub platform_version: String,
    pub max_iterations: usize,
    pub timeout_seconds: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            uncertainty_model: "bayesian_inference".to_string(),
            confidence_threshold: 0.75,
            verification_required: true,
            real_time_analysis: false,
            platform_version: "2.0.0-turbulance".to_string(),
            max_iterations: 10000,
            timeout_seconds: 300,
        }
    }
}

/// Execution context for maintaining state across analysis
#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    pub variables: HashMap<String, Value>,
    pub propositions: HashMap<String, PropositionState>,
    pub goals: HashMap<String, GoalState>,
    pub evidence: HashMap<String, EvidenceCollection>,
    pub session_id: Option<Uuid>,
    pub created_at: Option<DateTime<Utc>>,
}

impl TurbulanceEngine {
    /// Create a new Turbulance engine with default configuration
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
            context: ExecutionContext::default(),
        }
    }

    /// Create a new engine with custom configuration
    pub fn with_config(config: EngineConfig) -> Self {
        Self {
            config,
            context: ExecutionContext::default(),
        }
    }

    /// Parse Turbulance source code into an AST
    pub fn parse(&self, source: &str) -> Result<Program, TurbulanceError> {
        let tokens = TurbulanceLexer::tokenize(source)?;
        let ast = TurbulanceParser::parse_tokens(tokens)?;
        Ok(ast)
    }

    /// Compile AST into executable tasks
    pub fn compile(&self, ast: Program) -> Result<CompiledProgram, TurbulanceError> {
        let compiler = TurbulanceCompiler::new(&self.config);
        compiler.compile(ast)
    }

    /// Execute compiled program
    pub async fn execute(&mut self, program: CompiledProgram) -> Result<ExecutionResult, TurbulanceError> {
        let executor = TurbulanceExecutor::new(&self.config, &mut self.context);
        executor.execute(program).await
    }

    /// Parse, compile, and execute Turbulance code in one step
    pub async fn run(&mut self, source: &str) -> Result<ExecutionResult, TurbulanceError> {
        let ast = self.parse(source)?;
        let compiled = self.compile(ast)?;
        self.execute(compiled).await
    }
}

/// Result of executing a Turbulance program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub propositions: HashMap<String, PropositionResult>,
    pub goals: HashMap<String, GoalResult>,
    pub evidence: HashMap<String, EvidenceResult>,
    pub recommendations: Vec<Recommendation>,
    pub uncertainty_metrics: UncertaintyMetrics,
    pub execution_time: std::time::Duration,
    pub verification_results: Vec<VerificationResult>,
}

/// High-confidence recommendation generated from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub id: Uuid,
    pub description: String,
    pub confidence: f64,
    pub expected_impact: f64,
    pub implementation_difficulty: f64,
    pub safety_score: f64,
    pub evidence_strength: f64,
    pub recommendation_type: RecommendationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    TechniqueAdjustment,
    TrainingModification,
    InjuryPrevention,
    PerformanceOptimization,
    DataCollection,
    FurtherAnalysis,
}

/// Metrics for quantifying uncertainty across the analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyMetrics {
    pub overall_confidence: f64,
    pub evidence_reliability: f64,
    pub model_uncertainty: f64,
    pub data_quality: f64,
    pub prediction_variance: f64,
    pub bias_indicators: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_engine_creation() {
        let engine = TurbulanceEngine::new();
        assert_eq!(engine.config.confidence_threshold, 0.75);
        assert!(engine.config.verification_required);
    }

    #[tokio::test]
    async fn test_simple_parsing() {
        let engine = TurbulanceEngine::new();
        let source = r#"
            item test_var = 42
            funxn test_function():
                return test_var * 2
        "#;
        
        let result = engine.parse(source);
        assert!(result.is_ok());
    }
} 