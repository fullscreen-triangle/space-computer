//! # Turbulance Error Handling
//! 
//! Comprehensive error types and diagnostics for the Turbulance parser and compiler.

use thiserror::Error;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::lexer::TokenPosition;

/// All possible errors in the Turbulance parser and compiler
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum TurbulanceError {
    /// Lexical analysis errors
    #[error("Lexical error: {message}")]
    LexError {
        message: String,
        position: Option<TokenPosition>,
    },

    /// Syntax parsing errors
    #[error("Parse error: {message}")]
    ParseError {
        message: String,
        position: Option<TokenPosition>,
        expected: Option<Vec<String>>,
        found: Option<String>,
    },

    /// Semantic analysis errors
    #[error("Semantic error: {message}")]
    SemanticError {
        message: String,
        position: Option<TokenPosition>,
        error_type: SemanticErrorType,
    },

    /// Type checking errors
    #[error("Type error: {message}")]
    TypeError {
        message: String,
        position: Option<TokenPosition>,
        expected_type: Option<String>,
        found_type: Option<String>,
    },

    /// Compilation errors
    #[error("Compilation error: {message}")]
    CompilationError {
        message: String,
        position: Option<TokenPosition>,
        stage: CompilationStage,
    },

    /// Runtime execution errors
    #[error("Runtime error: {message}")]
    RuntimeError {
        message: String,
        position: Option<TokenPosition>,
        error_type: RuntimeErrorType,
    },

    /// Probabilistic reasoning errors
    #[error("Probabilistic error: {message}")]
    ProbabilisticError {
        message: String,
        confidence_level: Option<f64>,
        uncertainty_source: Option<String>,
    },

    /// Evidence integration errors
    #[error("Evidence error: {message}")]
    EvidenceError {
        message: String,
        evidence_source: Option<String>,
        quality_issue: Option<String>,
    },

    /// Goal optimization errors
    #[error("Goal optimization error: {message}")]
    GoalError {
        message: String,
        goal_id: Option<String>,
        convergence_issue: Option<String>,
    },

    /// Verification system errors
    #[error("Verification error: {message}")]
    VerificationError {
        message: String,
        verification_level: Option<String>,
        similarity_score: Option<f64>,
    },

    /// IO and system errors
    #[error("IO error: {message}")]
    IoError {
        message: String,
        file_path: Option<String>,
    },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    ConfigError {
        message: String,
        config_key: Option<String>,
    },

    /// Timeout errors
    #[error("Timeout error: {message}")]
    TimeoutError {
        message: String,
        timeout_duration: Option<std::time::Duration>,
    },

    /// Multiple errors collected together
    #[error("Multiple errors occurred")]
    MultipleErrors {
        errors: Vec<TurbulanceError>,
    },
}

/// Types of semantic errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticErrorType {
    UndefinedVariable,
    UndefinedFunction,
    DuplicateDefinition,
    InvalidScope,
    CircularReference,
    InvalidProposition,
    InvalidMotion,
    InvalidEvidence,
    InvalidGoal,
    ConstraintViolation,
}

/// Compilation stages where errors can occur
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompilationStage {
    LexicalAnalysis,
    SyntaxAnalysis,
    SemanticAnalysis,
    TypeChecking,
    CodeGeneration,
    Optimization,
    Linking,
}

/// Types of runtime errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeErrorType {
    DivisionByZero,
    IndexOutOfBounds,
    NullPointerAccess,
    StackOverflow,
    OutOfMemory,
    InvalidOperation,
    ConvergenceFailure,
    DataCorruption,
    NetworkError,
    TimeoutExpired,
}

/// Diagnostic information for errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub error: TurbulanceError,
    pub severity: DiagnosticSeverity,
    pub source_snippet: Option<String>,
    pub suggestions: Vec<String>,
    pub related_errors: Vec<TurbulanceError>,
    pub help_text: Option<String>,
}

/// Severity levels for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

impl TurbulanceError {
    /// Create a new lexical error
    pub fn lex_error(message: impl Into<String>, position: Option<TokenPosition>) -> Self {
        Self::LexError {
            message: message.into(),
            position,
        }
    }

    /// Create a new parse error
    pub fn parse_error(
        message: impl Into<String>,
        position: Option<TokenPosition>,
        expected: Option<Vec<String>>,
        found: Option<String>,
    ) -> Self {
        Self::ParseError {
            message: message.into(),
            position,
            expected,
            found,
        }
    }

    /// Create a new semantic error
    pub fn semantic_error(
        message: impl Into<String>,
        position: Option<TokenPosition>,
        error_type: SemanticErrorType,
    ) -> Self {
        Self::SemanticError {
            message: message.into(),
            position,
            error_type,
        }
    }

    /// Create a new type error
    pub fn type_error(
        message: impl Into<String>,
        position: Option<TokenPosition>,
        expected_type: Option<String>,
        found_type: Option<String>,
    ) -> Self {
        Self::TypeError {
            message: message.into(),
            position,
            expected_type,
            found_type,
        }
    }

    /// Create a new compilation error
    pub fn compilation_error(
        message: impl Into<String>,
        position: Option<TokenPosition>,
        stage: CompilationStage,
    ) -> Self {
        Self::CompilationError {
            message: message.into(),
            position,
            stage,
        }
    }

    /// Create a new runtime error
    pub fn runtime_error(
        message: impl Into<String>,
        position: Option<TokenPosition>,
        error_type: RuntimeErrorType,
    ) -> Self {
        Self::RuntimeError {
            message: message.into(),
            position,
            error_type,
        }
    }

    /// Create a new probabilistic error
    pub fn probabilistic_error(
        message: impl Into<String>,
        confidence_level: Option<f64>,
        uncertainty_source: Option<String>,
    ) -> Self {
        Self::ProbabilisticError {
            message: message.into(),
            confidence_level,
            uncertainty_source,
        }
    }

    /// Create a new evidence error
    pub fn evidence_error(
        message: impl Into<String>,
        evidence_source: Option<String>,
        quality_issue: Option<String>,
    ) -> Self {
        Self::EvidenceError {
            message: message.into(),
            evidence_source,
            quality_issue,
        }
    }

    /// Create a new goal error
    pub fn goal_error(
        message: impl Into<String>,
        goal_id: Option<String>,
        convergence_issue: Option<String>,
    ) -> Self {
        Self::GoalError {
            message: message.into(),
            goal_id,
            convergence_issue,
        }
    }

    /// Create a new verification error
    pub fn verification_error(
        message: impl Into<String>,
        verification_level: Option<String>,
        similarity_score: Option<f64>,
    ) -> Self {
        Self::VerificationError {
            message: message.into(),
            verification_level,
            similarity_score,
        }
    }

    /// Get the position information for this error, if available
    pub fn position(&self) -> Option<&TokenPosition> {
        match self {
            Self::LexError { position, .. }
            | Self::ParseError { position, .. }
            | Self::SemanticError { position, .. }
            | Self::TypeError { position, .. }
            | Self::CompilationError { position, .. }
            | Self::RuntimeError { position, .. } => position.as_ref(),
            _ => None,
        }
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::LexError { .. } => false,
            Self::ParseError { .. } => true,
            Self::SemanticError { error_type, .. } => match error_type {
                SemanticErrorType::UndefinedVariable => true,
                SemanticErrorType::UndefinedFunction => true,
                SemanticErrorType::DuplicateDefinition => false,
                SemanticErrorType::CircularReference => false,
                _ => true,
            },
            Self::TypeError { .. } => true,
            Self::CompilationError { .. } => false,
            Self::RuntimeError { error_type, .. } => match error_type {
                RuntimeErrorType::StackOverflow => false,
                RuntimeErrorType::OutOfMemory => false,
                RuntimeErrorType::DataCorruption => false,
                _ => true,
            },
            Self::ProbabilisticError { .. } => true,
            Self::EvidenceError { .. } => true,
            Self::GoalError { .. } => true,
            Self::VerificationError { .. } => true,
            Self::TimeoutError { .. } => true,
            _ => false,
        }
    }

    /// Get the severity level of this error
    pub fn severity(&self) -> DiagnosticSeverity {
        match self {
            Self::LexError { .. }
            | Self::ParseError { .. }
            | Self::SemanticError { .. }
            | Self::TypeError { .. }
            | Self::CompilationError { .. } => DiagnosticSeverity::Error,
            
            Self::RuntimeError { error_type, .. } => match error_type {
                RuntimeErrorType::StackOverflow 
                | RuntimeErrorType::OutOfMemory 
                | RuntimeErrorType::DataCorruption => DiagnosticSeverity::Error,
                _ => DiagnosticSeverity::Warning,
            },
            
            Self::ProbabilisticError { confidence_level, .. } => {
                if let Some(conf) = confidence_level {
                    if *conf < 0.5 {
                        DiagnosticSeverity::Warning
                    } else {
                        DiagnosticSeverity::Info
                    }
                } else {
                    DiagnosticSeverity::Warning
                }
            },
            
            Self::EvidenceError { .. } => DiagnosticSeverity::Warning,
            Self::GoalError { .. } => DiagnosticSeverity::Warning,
            Self::VerificationError { .. } => DiagnosticSeverity::Info,
            Self::ConfigError { .. } => DiagnosticSeverity::Error,
            Self::IoError { .. } => DiagnosticSeverity::Error,
            Self::TimeoutError { .. } => DiagnosticSeverity::Warning,
            Self::MultipleErrors { .. } => DiagnosticSeverity::Error,
        }
    }
}

/// Error collection and diagnostic utilities
pub struct ErrorCollector {
    errors: Vec<TurbulanceError>,
    warnings: Vec<TurbulanceError>,
    max_errors: usize,
}

impl ErrorCollector {
    /// Create a new error collector
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            max_errors: 100,
        }
    }

    /// Add an error to the collection
    pub fn add_error(&mut self, error: TurbulanceError) {
        match error.severity() {
            DiagnosticSeverity::Error => {
                if self.errors.len() < self.max_errors {
                    self.errors.push(error);
                }
            }
            DiagnosticSeverity::Warning => {
                if self.warnings.len() < self.max_errors {
                    self.warnings.push(error);
                }
            }
            _ => {
                // Info and hints are not collected by default
            }
        }
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get all errors
    pub fn errors(&self) -> &[TurbulanceError] {
        &self.errors
    }

    /// Get all warnings
    pub fn warnings(&self) -> &[TurbulanceError] {
        &self.warnings
    }

    /// Convert to a single error if there are multiple
    pub fn into_result(self) -> Result<(), TurbulanceError> {
        if self.errors.is_empty() {
            Ok(())
        } else if self.errors.len() == 1 {
            Err(self.errors.into_iter().next().unwrap())
        } else {
            Err(TurbulanceError::MultipleErrors {
                errors: self.errors,
            })
        }
    }
}

impl Default for ErrorCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SemanticErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemanticErrorType::UndefinedVariable => write!(f, "undefined variable"),
            SemanticErrorType::UndefinedFunction => write!(f, "undefined function"),
            SemanticErrorType::DuplicateDefinition => write!(f, "duplicate definition"),
            SemanticErrorType::InvalidScope => write!(f, "invalid scope"),
            SemanticErrorType::CircularReference => write!(f, "circular reference"),
            SemanticErrorType::InvalidProposition => write!(f, "invalid proposition"),
            SemanticErrorType::InvalidMotion => write!(f, "invalid motion"),
            SemanticErrorType::InvalidEvidence => write!(f, "invalid evidence"),
            SemanticErrorType::InvalidGoal => write!(f, "invalid goal"),
            SemanticErrorType::ConstraintViolation => write!(f, "constraint violation"),
        }
    }
}

impl fmt::Display for CompilationStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilationStage::LexicalAnalysis => write!(f, "lexical analysis"),
            CompilationStage::SyntaxAnalysis => write!(f, "syntax analysis"),
            CompilationStage::SemanticAnalysis => write!(f, "semantic analysis"),
            CompilationStage::TypeChecking => write!(f, "type checking"),
            CompilationStage::CodeGeneration => write!(f, "code generation"),
            CompilationStage::Optimization => write!(f, "optimization"),
            CompilationStage::Linking => write!(f, "linking"),
        }
    }
}

impl fmt::Display for RuntimeErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeErrorType::DivisionByZero => write!(f, "division by zero"),
            RuntimeErrorType::IndexOutOfBounds => write!(f, "index out of bounds"),
            RuntimeErrorType::NullPointerAccess => write!(f, "null pointer access"),
            RuntimeErrorType::StackOverflow => write!(f, "stack overflow"),
            RuntimeErrorType::OutOfMemory => write!(f, "out of memory"),
            RuntimeErrorType::InvalidOperation => write!(f, "invalid operation"),
            RuntimeErrorType::ConvergenceFailure => write!(f, "convergence failure"),
            RuntimeErrorType::DataCorruption => write!(f, "data corruption"),
            RuntimeErrorType::NetworkError => write!(f, "network error"),
            RuntimeErrorType::TimeoutExpired => write!(f, "timeout expired"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = TurbulanceError::lex_error("Test message", None);
        assert!(matches!(error, TurbulanceError::LexError { .. }));
        assert_eq!(error.severity(), DiagnosticSeverity::Error);
    }

    #[test]
    fn test_error_collector() {
        let mut collector = ErrorCollector::new();
        
        collector.add_error(TurbulanceError::lex_error("Error 1", None));
        collector.add_error(TurbulanceError::probabilistic_error("Warning 1", Some(0.3), None));
        
        assert!(collector.has_errors());
        assert_eq!(collector.errors().len(), 1);
        assert_eq!(collector.warnings().len(), 1);
    }

    #[test]
    fn test_recoverable_errors() {
        let recoverable = TurbulanceError::parse_error("Parse error", None, None, None);
        let non_recoverable = TurbulanceError::lex_error("Lex error", None);
        
        assert!(recoverable.is_recoverable());
        assert!(!non_recoverable.is_recoverable());
    }

    #[test]
    fn test_multiple_errors() {
        let errors = vec![
            TurbulanceError::lex_error("Error 1", None),
            TurbulanceError::parse_error("Error 2", None, None, None),
        ];
        
        let multiple = TurbulanceError::MultipleErrors { errors: errors.clone() };
        
        match multiple {
            TurbulanceError::MultipleErrors { errors: collected } => {
                assert_eq!(collected.len(), 2);
            }
            _ => panic!("Expected MultipleErrors"),
        }
    }
} 