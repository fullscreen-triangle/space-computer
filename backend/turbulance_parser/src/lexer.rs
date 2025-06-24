//! # Turbulance Lexer
//! 
//! Tokenizes Turbulance source code using the Logos crate for efficient lexing.

use logos::Logos;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::TurbulanceError;

/// All possible tokens in the Turbulance language
#[derive(Logos, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Token {
    // Keywords
    #[token("item")]
    Item,
    #[token("funxn")]
    Function,
    #[token("given")]
    Given,
    #[token("within")]
    Within,
    #[token("considering")]
    Considering,
    #[token("proposition")]
    Proposition,
    #[token("motion")]
    Motion,
    #[token("evidence")]
    Evidence,
    #[token("metacognitive")]
    Metacognitive,
    #[token("goal")]
    Goal,
    #[token("config")]
    Config,
    #[token("datasources")]
    Datasources,
    #[token("evidence_integrator")]
    EvidenceIntegrator,
    #[token("orchestrator")]
    Orchestrator,
    #[token("verification_system")]
    VerificationSystem,
    #[token("real_time_orchestrator")]
    RealTimeOrchestrator,
    #[token("interface")]
    Interface,
    #[token("temporal")]
    Temporal,
    #[token("cross_domain_analysis")]
    CrossDomainAnalysis,
    #[token("pattern_registry")]
    PatternRegistry,
    
    // Control flow
    #[token("for")]
    For,
    #[token("each")]
    Each,
    #[token("in")]
    In,
    #[token("while")]
    While,
    #[token("if")]
    If,
    #[token("otherwise")]
    Otherwise,
    #[token("return")]
    Return,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("try")]
    Try,
    #[token("catch")]
    Catch,
    #[token("finally")]
    Finally,
    #[token("every")]
    Every,
    #[token("phase")]
    Phase,
    #[token("parallel")]
    Parallel,
    #[token("async")]
    Async,
    #[token("await")]
    Await,
    #[token("stream")]
    Stream,
    #[token("segment")]
    Segment,
    
    // Evidence and probabilistic constructs
    #[token("support")]
    Support,
    #[token("contradict")]
    Contradict,
    #[token("with_confidence")]
    WithConfidence,
    #[token("with_weight")]
    WithWeight,
    #[token("with_uncertainty")]
    WithUncertainty,
    #[token("bayesian_update")]
    BayesianUpdate,
    #[token("monte_carlo_simulation")]
    MonteCarloSimulation,
    #[token("uncertainty_propagation")]
    UncertaintyPropagation,
    #[token("confidence_threshold")]
    ConfidenceThreshold,
    
    // Goal-specific keywords
    #[token("success_threshold")]
    SuccessThreshold,
    #[token("metrics")]
    Metrics,
    #[token("objectives")]
    Objectives,
    #[token("constraints")]
    Constraints,
    #[token("optimization_algorithm")]
    OptimizationAlgorithm,
    #[token("personalization_factors")]
    PersonalizationFactors,
    #[token("adaptation_strategy")]
    AdaptationStrategy,
    
    // Evidence keywords
    #[token("sources")]
    Sources,
    #[token("collection")]
    Collection,
    #[token("processing")]
    Processing,
    #[token("validation")]
    Validation,
    #[token("fusion_methods")]
    FusionMethods,
    #[token("validation_pipeline")]
    ValidationPipeline,
    
    // Metacognitive keywords
    #[token("track")]
    Track,
    #[token("evaluate")]
    Evaluate,
    #[token("adapt")]
    Adapt,
    #[token("monitor")]
    Monitor,
    
    // Verification keywords
    #[token("verification_methods")]
    VerificationMethods,
    #[token("verification_levels")]
    VerificationLevels,
    
    // Temporal keywords
    #[token("scope")]
    Scope,
    #[token("patterns")]
    Patterns,
    #[token("operations")]
    Operations,
    
    // Context keywords
    #[token("context")]
    Context,
    #[token("extends")]
    Extends,
    
    // Logical operators
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("not")]
    Not,
    
    // Comparison operators
    #[token("==")]
    Equal,
    #[token("!=")]
    NotEqual,
    #[token("<")]
    Less,
    #[token("<=")]
    LessEqual,
    #[token(">")]
    Greater,
    #[token(">=")]
    GreaterEqual,
    
    // Arithmetic operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Multiply,
    #[token("/")]
    Divide,
    #[token("%")]
    Modulo,
    #[token("**")]
    Power,
    
    // Assignment operators
    #[token("=")]
    Assign,
    #[token("+=")]
    AddAssign,
    #[token("-=")]
    SubAssign,
    #[token("*=")]
    MulAssign,
    #[token("/=")]
    DivAssign,
    
    // Delimiters
    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token(".")]
    Dot,
    #[token("->")]
    Arrow,
    #[token("&&")]
    LogicalAnd,
    #[token("||")]
    LogicalOr,
    
    // Special operators
    #[token("±")]
    PlusMinus,
    #[token("°")]
    Degree,
    #[token("&")]
    Ampersand,
    #[token("|")]
    Pipe,
    #[token("?")]
    Question,
    #[token("..")]
    DotDot,
    #[token("...")]
    DotDotDot,
    
    // Literals
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),
    
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Integer(i64),
    
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        Some(s[1..s.len()-1].to_string())
    })]
    String(String),
    
    #[regex(r"'([^'\\]|\\.)*'", |lex| {
        let s = lex.slice();
        Some(s[1..s.len()-1].to_string())
    })]
    Char(String),
    
    #[token("true", |_| true)]
    #[token("false", |_| false)]
    Boolean(bool),
    
    // Identifiers (must come after keywords)
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),
    
    // Comments
    #[regex(r"//[^\r\n]*", logos::skip)]
    LineComment,
    
    #[regex(r"/\*([^*]|\*[^/])*\*/", logos::skip)]
    BlockComment,
    
    // Whitespace
    #[regex(r"[ \t\f]+", logos::skip)]
    Whitespace,
    
    #[regex(r"\r?\n", logos::skip)]
    Newline,
    
    // Error handling
    #[error]
    Error,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Item => write!(f, "item"),
            Token::Function => write!(f, "funxn"),
            Token::Given => write!(f, "given"),
            Token::Within => write!(f, "within"),
            Token::Considering => write!(f, "considering"),
            Token::Proposition => write!(f, "proposition"),
            Token::Motion => write!(f, "motion"),
            Token::Evidence => write!(f, "evidence"),
            Token::Metacognitive => write!(f, "metacognitive"),
            Token::Goal => write!(f, "goal"),
            Token::Support => write!(f, "support"),
            Token::Contradict => write!(f, "contradict"),
            Token::WithConfidence => write!(f, "with_confidence"),
            Token::WithWeight => write!(f, "with_weight"),
            Token::Identifier(s) => write!(f, "{}", s),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Integer(i) => write!(f, "{}", i),
            Token::Float(fl) => write!(f, "{}", fl),
            Token::Boolean(b) => write!(f, "{}", b),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Position information for tokens
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenPosition {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
    pub length: usize,
}

/// Token with position information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionedToken {
    pub token: Token,
    pub position: TokenPosition,
    pub lexeme: String,
}

/// Turbulance lexer that maintains position information
pub struct TurbulanceLexer {
    source: String,
    tokens: Vec<PositionedToken>,
}

impl TurbulanceLexer {
    /// Create a new lexer
    pub fn new(source: String) -> Self {
        Self {
            source,
            tokens: Vec::new(),
        }
    }
    
    /// Tokenize the source code
    pub fn tokenize(source: &str) -> Result<Vec<PositionedToken>, TurbulanceError> {
        let mut lexer = Self::new(source.to_string());
        lexer.lex()?;
        Ok(lexer.tokens)
    }
    
    /// Perform lexical analysis
    fn lex(&mut self) -> Result<(), TurbulanceError> {
        let mut lex = Token::lexer(&self.source);
        let mut line = 1;
        let mut line_start = 0;
        
        while let Some(token) = lex.next() {
            let span = lex.span();
            let lexeme = lex.slice().to_string();
            
            // Calculate position information
            let lines_before = self.source[..span.start].matches('\n').count();
            if lines_before > 0 {
                line += lines_before;
                line_start = self.source[..span.start].rfind('\n').unwrap_or(0) + 1;
            }
            
            let position = TokenPosition {
                line,
                column: span.start - line_start + 1,
                offset: span.start,
                length: span.len(),
            };
            
            match token {
                Token::Error => {
                    return Err(TurbulanceError::LexError {
                        message: format!("Unexpected character at line {}, column {}", position.line, position.column),
                        position: Some(position),
                    });
                }
                token => {
                    self.tokens.push(PositionedToken {
                        token,
                        position,
                        lexeme,
                    });
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let source = "item x = 42";
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].token, Token::Item);
        assert_eq!(tokens[1].token, Token::Identifier("x".to_string()));
        assert_eq!(tokens[2].token, Token::Assign);
        assert_eq!(tokens[3].token, Token::Integer(42));
    }

    #[test]
    fn test_proposition_tokenization() {
        let source = r#"
            proposition TestProp:
                motion TestMotion("description")
                given condition with_confidence(0.8):
                    support TestMotion with_weight(0.9)
        "#;
        
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        
        // Verify key tokens are present
        assert!(tokens.iter().any(|t| matches!(t.token, Token::Proposition)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::Motion)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::Given)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::Support)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::WithConfidence)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::WithWeight)));
    }

    #[test]
    fn test_probabilistic_constructs() {
        let source = "uncertainty_propagation bayesian_update monte_carlo_simulation";
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].token, Token::UncertaintyPropagation);
        assert_eq!(tokens[1].token, Token::BayesianUpdate);
        assert_eq!(tokens[2].token, Token::MonteCarloSimulation);
    }

    #[test]
    fn test_numeric_literals() {
        let source = "42 3.14 0.95 ± 0.03";
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        
        assert_eq!(tokens[0].token, Token::Integer(42));
        assert_eq!(tokens[1].token, Token::Float(3.14));
        assert_eq!(tokens[2].token, Token::Float(0.95));
        assert_eq!(tokens[3].token, Token::PlusMinus);
        assert_eq!(tokens[4].token, Token::Float(0.03));
    }

    #[test]
    fn test_string_literals() {
        let source = r#""test string" 'c'"#;
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        
        assert_eq!(tokens[0].token, Token::String("test string".to_string()));
        assert_eq!(tokens[1].token, Token::Char("c".to_string()));
    }

    #[test]
    fn test_position_tracking() {
        let source = "line1\nline2\nline3";
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        
        assert_eq!(tokens[0].position.line, 1);
        assert_eq!(tokens[1].position.line, 2);
        assert_eq!(tokens[2].position.line, 3);
    }

    #[test]
    fn test_error_handling() {
        let source = "valid @invalid";
        let result = TurbulanceLexer::tokenize(source);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            TurbulanceError::LexError { message, position } => {
                assert!(message.contains("Unexpected character"));
                assert!(position.is_some());
            }
            _ => panic!("Expected LexError"),
        }
    }
} 