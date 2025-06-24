//! # Turbulance Parser
//! 
//! Recursive descent parser for the Turbulance probabilistic scientific programming language.

use std::collections::HashMap;
use crate::{
    ast::*,
    error::{TurbulanceError, SemanticErrorType},
    lexer::{Token, PositionedToken, TokenPosition},
};

/// Main parser for Turbulance language
pub struct TurbulanceParser {
    tokens: Vec<PositionedToken>,
    current: usize,
}

impl TurbulanceParser {
    /// Create a new parser with tokens
    pub fn new(tokens: Vec<PositionedToken>) -> Self {
        Self { tokens, current: 0 }
    }

    /// Parse tokens into a Program AST
    pub fn parse_tokens(tokens: Vec<PositionedToken>) -> Result<Program, TurbulanceError> {
        let mut parser = Self::new(tokens);
        parser.parse_program()
    }

    /// Parse a complete program
    pub fn parse_program(&mut self) -> Result<Program, TurbulanceError> {
        let mut program = Program::new();
        
        while !self.is_at_end() {
            match self.parse_item() {
                Ok(item) => program.add_item(item),
                Err(e) => {
                    if !e.is_recoverable() {
                        return Err(e);
                    }
                    // Try to recover from error by advancing to next valid token
                    self.synchronize();
                }
            }
        }
        
        Ok(program)
    }

    /// Parse a top-level item
    fn parse_item(&mut self) -> Result<Item, TurbulanceError> {
        match self.peek_token() {
            Some(token) => match &token.token {
                Token::Item => self.parse_variable_declaration(),
                Token::Function => self.parse_function_declaration(),
                Token::Proposition => self.parse_proposition_declaration(),
                Token::Goal => self.parse_goal_declaration(),
                Token::Evidence => self.parse_evidence_declaration(),
                Token::Metacognitive => self.parse_metacognitive_declaration(),
                Token::Config => self.parse_config_declaration(),
                Token::Datasources => self.parse_datasources_declaration(),
                Token::EvidenceIntegrator => self.parse_evidence_integrator_declaration(),
                Token::Orchestrator => self.parse_orchestrator_declaration(),
                _ => {
                    let expr = self.parse_expression()?;
                    Ok(Item::Expression(expr))
                }
            },
            None => Err(TurbulanceError::parse_error(
                "Unexpected end of file",
                None,
                None,
                None,
            ))
        }
    }

    /// Parse variable declaration: `item name = value`
    fn parse_variable_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::Item, "Expected 'item'")?;
        
        let name = self.parse_identifier()?;
        let type_annotation = if self.match_token(&Token::Colon) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        
        self.consume(Token::Assign, "Expected '='")?;
        let value = self.parse_expression()?;
        
        // Check for uncertainty specification
        let uncertainty = if self.match_token(&Token::WithUncertainty) {
            Some(self.parse_uncertainty_spec()?)
        } else {
            None
        };

        Ok(Item::Variable(VariableDeclaration {
            name,
            type_annotation,
            value,
            uncertainty,
            position: self.previous_position(),
        }))
    }

    /// Parse function declaration: `funxn name(params): body`
    fn parse_function_declaration(&mut self) -> Result<Item, TurbulanceError> {
        let is_async = self.match_token(&Token::Async);
        self.consume(Token::Function, "Expected 'funxn'")?;
        
        let name = self.parse_identifier()?;
        
        self.consume(Token::LeftParen, "Expected '('")?;
        let parameters = self.parse_parameter_list()?;
        self.consume(Token::RightParen, "Expected ')'")?;
        
        let return_type = if self.match_token(&Token::Arrow) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        
        self.consume(Token::Colon, "Expected ':'")?;
        let body = self.parse_statement_block()?;
        
        // Check for uncertainty propagation specification
        let uncertainty_propagation = if self.match_token(&Token::UncertaintyPropagation) {
            Some(self.parse_uncertainty_propagation()?)
        } else {
            None
        };

        Ok(Item::Function(FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
            is_async,
            uncertainty_propagation,
            position: self.previous_position(),
        }))
    }

    /// Parse proposition declaration
    fn parse_proposition_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::Proposition, "Expected 'proposition'")?;
        let name = self.parse_identifier()?;
        
        // Check for inheritance
        let extends = if self.match_token(&Token::Extends) {
            Some(self.parse_identifier()?)
        } else {
            None
        };
        
        self.consume(Token::Colon, "Expected ':'")?;
        
        let mut context = Vec::new();
        let mut motions = Vec::new();
        let mut evidence_evaluation = Vec::new();
        let mut confidence_threshold = None;
        
        // Parse proposition body
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if self.match_token(&Token::Context) {
                context.push(self.parse_context_declaration()?);
            } else if self.match_token(&Token::Motion) {
                motions.push(self.parse_motion_declaration()?);
            } else if self.match_token(&Token::Within) {
                evidence_evaluation.push(self.parse_evidence_evaluation()?);
            } else if self.match_token(&Token::ConfidenceThreshold) {
                self.consume(Token::Assign, "Expected '='")?;
                confidence_threshold = Some(self.parse_float_literal()?);
            } else {
                return Err(TurbulanceError::parse_error(
                    "Expected 'context', 'motion', 'within', or 'confidence_threshold'",
                    self.current_position(),
                    None,
                    None,
                ));
            }
        }

        Ok(Item::Proposition(PropositionDeclaration {
            name,
            extends,
            context,
            motions,
            evidence_evaluation,
            confidence_threshold,
            position: self.previous_position(),
        }))
    }

    /// Parse goal declaration
    fn parse_goal_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::Goal, "Expected 'goal'")?;
        let name = self.parse_identifier()?;
        self.consume(Token::Assign, "Expected '='")?;
        
        // Parse Goal.new(...) call
        let goal_expr = self.parse_expression()?;
        
        // For now, extract goal properties from expression
        // In a full implementation, this would be more sophisticated
        let (description, objectives, success_thresholds) = self.extract_goal_properties(&goal_expr)?;

        Ok(Item::Goal(GoalDeclaration {
            name,
            description,
            objectives,
            success_thresholds,
            optimization_algorithm: None,
            personalization_factors: None,
            adaptation_strategy: None,
            sub_goals: Vec::new(),
            position: self.previous_position(),
        }))
    }

    /// Parse evidence declaration
    fn parse_evidence_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::Evidence, "Expected 'evidence'")?;
        let name = self.parse_identifier()?;
        self.consume(Token::Colon, "Expected ':'")?;
        
        let mut sources = Vec::new();
        let mut collection_strategy = None;
        let mut processing_pipeline = None;
        let mut validation_rules = None;
        let mut quality_metrics = None;
        
        // Parse evidence body
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if self.match_token(&Token::Sources) {
                self.consume(Token::Colon, "Expected ':'")?;
                sources = self.parse_evidence_sources()?;
            } else if self.match_token(&Token::Collection) {
                self.consume(Token::Colon, "Expected ':'")?;
                collection_strategy = Some(self.parse_collection_strategy()?);
            } else if self.match_token(&Token::Processing) {
                self.consume(Token::Colon, "Expected ':'")?;
                processing_pipeline = Some(self.parse_processing_pipeline()?);
            } else if self.match_token(&Token::Validation) {
                self.consume(Token::Colon, "Expected ':'")?;
                validation_rules = Some(self.parse_validation_rules()?);
            } else {
                self.advance(); // Skip unknown tokens for now
            }
        }

        Ok(Item::Evidence(EvidenceDeclaration {
            name,
            sources,
            collection_strategy,
            processing_pipeline,
            validation_rules,
            quality_metrics,
            position: self.previous_position(),
        }))
    }

    /// Parse metacognitive declaration
    fn parse_metacognitive_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::Metacognitive, "Expected 'metacognitive'")?;
        let name = self.parse_identifier()?;
        self.consume(Token::Colon, "Expected ':'")?;
        
        let mut tracking_dimensions = Vec::new();
        let mut evaluation_methods = Vec::new();
        let mut adaptation_rules = Vec::new();
        let mut quality_assurance = None;
        
        // Parse metacognitive body
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if self.match_token(&Token::Track) {
                self.consume(Token::Colon, "Expected ':'")?;
                tracking_dimensions = self.parse_tracking_dimensions()?;
            } else if self.match_token(&Token::Evaluate) {
                self.consume(Token::Colon, "Expected ':'")?;
                evaluation_methods = self.parse_evaluation_methods()?;
            } else if self.match_token(&Token::Adapt) {
                self.consume(Token::Colon, "Expected ':'")?;
                adaptation_rules = self.parse_adaptation_rules()?;
            } else {
                self.advance(); // Skip unknown tokens for now
            }
        }

        Ok(Item::Metacognitive(MetacognitiveDeclaration {
            name,
            tracking_dimensions,
            evaluation_methods,
            adaptation_rules,
            quality_assurance,
            position: self.previous_position(),
        }))
    }

    /// Parse config declaration
    fn parse_config_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::Config, "Expected 'config'")?;
        let name = self.parse_identifier()?;
        self.consume(Token::Colon, "Expected ':'")?;
        
        let settings = self.parse_object_literal()?;

        Ok(Item::Config(ConfigDeclaration {
            name,
            settings,
            position: self.previous_position(),
        }))
    }

    /// Parse datasources declaration
    fn parse_datasources_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::Datasources, "Expected 'datasources'")?;
        let name = self.parse_identifier()?;
        self.consume(Token::Colon, "Expected ':'")?;
        
        let sources = self.parse_datasource_specs()?;

        Ok(Item::Datasources(DatasourcesDeclaration {
            name,
            sources,
            position: self.previous_position(),
        }))
    }

    /// Parse evidence integrator declaration
    fn parse_evidence_integrator_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::EvidenceIntegrator, "Expected 'evidence_integrator'")?;
        let name = self.parse_identifier()?;
        self.consume(Token::Colon, "Expected ':'")?;
        
        let mut sources = Vec::new();
        let mut fusion_methods = Vec::new();
        let mut validation_pipeline = Vec::new();
        
        // Parse integrator body
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if self.match_token(&Token::Sources) {
                self.consume(Token::Colon, "Expected ':'")?;
                sources = self.parse_evidence_sources()?;
            } else if self.match_token(&Token::FusionMethods) {
                self.consume(Token::Colon, "Expected ':'")?;
                fusion_methods = self.parse_fusion_methods()?;
            } else if self.match_token(&Token::ValidationPipeline) {
                self.consume(Token::Colon, "Expected ':'")?;
                validation_pipeline = self.parse_validation_steps()?;
            } else {
                self.advance(); // Skip unknown tokens for now
            }
        }

        Ok(Item::EvidenceIntegrator(EvidenceIntegratorDeclaration {
            name,
            sources,
            fusion_methods,
            validation_pipeline,
            position: self.previous_position(),
        }))
    }

    /// Parse orchestrator declaration
    fn parse_orchestrator_declaration(&mut self) -> Result<Item, TurbulanceError> {
        self.consume(Token::Orchestrator, "Expected 'orchestrator'")?;
        let name = self.parse_identifier()?;
        self.consume(Token::Colon, "Expected ':'")?;
        
        let mut initialization_steps = Vec::new();
        let mut execution_phases = Vec::new();
        let mut monitoring_strategy = None;
        
        // Parse orchestrator body - simplified for now
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            self.advance(); // Skip for now
        }

        Ok(Item::Orchestrator(OrchestratorDeclaration {
            name,
            initialization_steps,
            execution_phases,
            monitoring_strategy,
            position: self.previous_position(),
        }))
    }

    /// Helper parsing methods
    fn parse_identifier(&mut self) -> Result<Identifier, TurbulanceError> {
        if let Some(token) = self.peek_token() {
            if let Token::Identifier(name) = &token.token {
                let identifier = Identifier::with_position(name.clone(), token.position.clone());
                self.advance();
                Ok(identifier)
            } else {
                Err(TurbulanceError::parse_error(
                    "Expected identifier",
                    Some(token.position.clone()),
                    Some(vec!["identifier".to_string()]),
                    Some(format!("{}", token.token)),
                ))
            }
        } else {
            Err(TurbulanceError::parse_error(
                "Expected identifier, found end of file",
                None,
                None,
                None,
            ))
        }
    }

    fn parse_parameter_list(&mut self) -> Result<Vec<Parameter>, TurbulanceError> {
        let mut parameters = Vec::new();
        
        if !self.check(&Token::RightParen) {
            loop {
                let name = self.parse_identifier()?;
                let type_annotation = if self.match_token(&Token::Colon) {
                    Some(self.parse_type_annotation()?)
                } else {
                    None
                };
                
                let default_value = if self.match_token(&Token::Assign) {
                    Some(self.parse_expression()?)
                } else {
                    None
                };
                
                parameters.push(Parameter {
                    name,
                    type_annotation,
                    default_value,
                    confidence_requirement: None,
                });
                
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        Ok(parameters)
    }

    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, TurbulanceError> {
        let type_name = if let Some(token) = self.peek_token() {
            if let Token::Identifier(name) = &token.token {
                self.advance();
                name.clone()
            } else {
                return Err(TurbulanceError::parse_error(
                    "Expected type name",
                    Some(token.position.clone()),
                    None,
                    None,
                ));
            }
        } else {
            return Err(TurbulanceError::parse_error(
                "Expected type name",
                None,
                None,
                None,
            ));
        };

        Ok(TypeAnnotation {
            type_name,
            generic_params: None,
            optional: false,
        })
    }

    fn parse_statement_block(&mut self) -> Result<Vec<Statement>, TurbulanceError> {
        let mut statements = Vec::new();
        
        // For now, parse a simple expression as statement
        // In full implementation, this would handle indentation and complex blocks
        if !self.is_at_end() {
            let expr = self.parse_expression()?;
            statements.push(Statement::Expression(expr));
        }
        
        Ok(statements)
    }

    fn parse_expression(&mut self) -> Result<Expression, TurbulanceError> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<Expression, TurbulanceError> {
        let mut expr = self.parse_logical_and()?;
        
        while self.match_token(&Token::Or) || self.match_token(&Token::LogicalOr) {
            let operator = BinaryOperator::Or;
            let right = self.parse_logical_and()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_logical_and(&mut self) -> Result<Expression, TurbulanceError> {
        let mut expr = self.parse_equality()?;
        
        while self.match_token(&Token::And) || self.match_token(&Token::LogicalAnd) {
            let operator = BinaryOperator::And;
            let right = self.parse_equality()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expression, TurbulanceError> {
        let mut expr = self.parse_comparison()?;
        
        while let Some(token) = self.peek_token() {
            let operator = match &token.token {
                Token::Equal => BinaryOperator::Equal,
                Token::NotEqual => BinaryOperator::NotEqual,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_comparison()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expression, TurbulanceError> {
        let mut expr = self.parse_term()?;
        
        while let Some(token) = self.peek_token() {
            let operator = match &token.token {
                Token::Greater => BinaryOperator::Greater,
                Token::GreaterEqual => BinaryOperator::GreaterEqual,
                Token::Less => BinaryOperator::Less,
                Token::LessEqual => BinaryOperator::LessEqual,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_term()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expression, TurbulanceError> {
        let mut expr = self.parse_factor()?;
        
        while let Some(token) = self.peek_token() {
            let operator = match &token.token {
                Token::Plus => BinaryOperator::Add,
                Token::Minus => BinaryOperator::Subtract,
                Token::PlusMinus => BinaryOperator::PlusMinus,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_factor()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expression, TurbulanceError> {
        let mut expr = self.parse_unary()?;
        
        while let Some(token) = self.peek_token() {
            let operator = match &token.token {
                Token::Multiply => BinaryOperator::Multiply,
                Token::Divide => BinaryOperator::Divide,
                Token::Modulo => BinaryOperator::Modulo,
                Token::Power => BinaryOperator::Power,
                _ => break,
            };
            
            self.advance();
            let right = self.parse_unary()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expression, TurbulanceError> {
        if let Some(token) = self.peek_token() {
            let operator = match &token.token {
                Token::Not => UnaryOperator::Not,
                Token::Minus => UnaryOperator::Minus,
                Token::Plus => UnaryOperator::Plus,
                _ => return self.parse_call(),
            };
            
            self.advance();
            let operand = self.parse_unary()?;
            Ok(Expression::UnaryOp {
                operator,
                operand: Box::new(operand),
            })
        } else {
            Err(TurbulanceError::parse_error(
                "Unexpected end of file in unary expression",
                None,
                None,
                None,
            ))
        }
    }

    fn parse_call(&mut self) -> Result<Expression, TurbulanceError> {
        let mut expr = self.parse_primary()?;
        
        loop {
            if self.match_token(&Token::LeftParen) {
                expr = self.finish_call(expr)?;
            } else if self.match_token(&Token::Dot) {
                let name = self.parse_identifier()?;
                if self.match_token(&Token::LeftParen) {
                    let arguments = self.parse_argument_list()?;
                    self.consume(Token::RightParen, "Expected ')'")?;
                    expr = Expression::MethodCall {
                        object: Box::new(expr),
                        method: name.name,
                        arguments,
                    };
                } else {
                    expr = Expression::Field {
                        object: Box::new(expr),
                        field: name.name,
                    };
                }
            } else if self.match_token(&Token::LeftBracket) {
                let index = self.parse_expression()?;
                self.consume(Token::RightBracket, "Expected ']'")?;
                expr = Expression::Index {
                    object: Box::new(expr),
                    index: Box::new(index),
                };
            } else {
                break;
            }
        }
        
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expression, TurbulanceError> {
        if let Some(token) = self.peek_token() {
            match &token.token {
                Token::Integer(value) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Integer(*value)))
                }
                Token::Float(value) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Float(*value)))
                }
                Token::String(value) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::String(value.clone())))
                }
                Token::Boolean(value) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Boolean(*value)))
                }
                Token::Identifier(name) => {
                    let identifier = Identifier::with_position(name.clone(), token.position.clone());
                    self.advance();
                    Ok(Expression::Identifier(identifier))
                }
                Token::LeftParen => {
                    self.advance();
                    let expr = self.parse_expression()?;
                    self.consume(Token::RightParen, "Expected ')'")?;
                    Ok(expr)
                }
                Token::LeftBracket => {
                    self.advance();
                    let elements = self.parse_array_elements()?;
                    self.consume(Token::RightBracket, "Expected ']'")?;
                    Ok(Expression::Array(elements))
                }
                Token::LeftBrace => {
                    self.advance();
                    let object = self.parse_object_literal()?;
                    self.consume(Token::RightBrace, "Expected '}'")?;
                    Ok(Expression::Object(object))
                }
                _ => Err(TurbulanceError::parse_error(
                    "Expected expression",
                    Some(token.position.clone()),
                    None,
                    Some(format!("{}", token.token)),
                ))
            }
        } else {
            Err(TurbulanceError::parse_error(
                "Expected expression, found end of file",
                None,
                None,
                None,
            ))
        }
    }

    // Placeholder implementations for complex parsing methods
    fn parse_context_declaration(&mut self) -> Result<ContextDeclaration, TurbulanceError> {
        let name = self.parse_identifier()?;
        self.consume(Token::Assign, "Expected '='")?;
        let value = self.parse_expression()?;
        
        Ok(ContextDeclaration {
            name,
            value,
            uncertainty: None,
        })
    }

    fn parse_motion_declaration(&mut self) -> Result<MotionDeclaration, TurbulanceError> {
        let name = self.parse_identifier()?;
        self.consume(Token::LeftParen, "Expected '('")?;
        let description = self.parse_string_literal()?;
        self.consume(Token::RightParen, "Expected ')'")?;
        
        Ok(MotionDeclaration {
            name,
            description,
            requirements: None,
            criteria: None,
            patterns: None,
            confidence_threshold: None,
            position: self.previous_position(),
        })
    }

    fn parse_evidence_evaluation(&mut self) -> Result<EvidenceEvaluation, TurbulanceError> {
        // Parse "within scope:" clause
        let scope_expression = self.parse_expression()?;
        let scope = WithinClause {
            scope_expression,
            alias: None,
        };
        
        self.consume(Token::Colon, "Expected ':'")?;
        
        // Parse given clauses
        let mut conditions = Vec::new();
        while self.match_token(&Token::Given) {
            conditions.push(self.parse_given_clause()?);
        }
        
        Ok(EvidenceEvaluation {
            scope,
            conditions,
            position: self.previous_position(),
        })
    }

    fn parse_given_clause(&mut self) -> Result<GivenClause, TurbulanceError> {
        let condition = self.parse_expression()?;
        
        let confidence = if self.match_token(&Token::WithConfidence) {
            Some(ConfidenceSpec::Value(self.parse_float_literal()?))
        } else {
            None
        };
        
        self.consume(Token::Colon, "Expected ':'")?;
        
        let mut actions = Vec::new();
        if self.match_token(&Token::Support) {
            let motion = self.parse_identifier()?;
            let weight = if self.match_token(&Token::WithWeight) {
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            actions.push(EvidenceAction::Support {
                motion,
                weight,
                confidence: None,
            });
        } else if self.match_token(&Token::Contradict) {
            let motion = self.parse_identifier()?;
            let weight = if self.match_token(&Token::WithWeight) {
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            actions.push(EvidenceAction::Contradict {
                motion,
                weight,
                confidence: None,
            });
        }
        
        Ok(GivenClause {
            condition,
            confidence,
            actions,
        })
    }

    // More placeholder implementations
    fn parse_uncertainty_spec(&mut self) -> Result<UncertaintySpec, TurbulanceError> {
        // Simplified uncertainty parsing
        let value = self.parse_float_literal()?;
        Ok(UncertaintySpec::Range(value))
    }

    fn parse_uncertainty_propagation(&mut self) -> Result<UncertaintyPropagation, TurbulanceError> {
        // Simplified propagation parsing
        Ok(UncertaintyPropagation::Linear)
    }

    fn extract_goal_properties(&self, _expr: &Expression) -> Result<(String, Vec<Objective>, HashMap<String, ThresholdSpec>), TurbulanceError> {
        // Simplified goal property extraction
        Ok((
            "Default goal description".to_string(),
            Vec::new(),
            HashMap::new(),
        ))
    }

    // Additional placeholder methods for evidence, metacognitive, etc.
    fn parse_evidence_sources(&mut self) -> Result<Vec<EvidenceSource>, TurbulanceError> {
        Ok(Vec::new()) // Placeholder
    }

    fn parse_collection_strategy(&mut self) -> Result<CollectionStrategy, TurbulanceError> {
        Ok(CollectionStrategy {
            frequency: None,
            duration: None,
            validation_method: None,
            quality_threshold: None,
        })
    }

    fn parse_processing_pipeline(&mut self) -> Result<Vec<ProcessingStep>, TurbulanceError> {
        Ok(Vec::new()) // Placeholder
    }

    fn parse_validation_rules(&mut self) -> Result<Vec<ValidationRule>, TurbulanceError> {
        Ok(Vec::new()) // Placeholder
    }

    fn parse_tracking_dimensions(&mut self) -> Result<Vec<TrackingDimension>, TurbulanceError> {
        Ok(Vec::new()) // Placeholder
    }

    fn parse_evaluation_methods(&mut self) -> Result<Vec<EvaluationMethod>, TurbulanceError> {
        Ok(Vec::new()) // Placeholder
    }

    fn parse_adaptation_rules(&mut self) -> Result<Vec<AdaptationRule>, TurbulanceError> {
        Ok(Vec::new()) // Placeholder
    }

    fn parse_datasource_specs(&mut self) -> Result<HashMap<String, DataSourceSpec>, TurbulanceError> {
        Ok(HashMap::new()) // Placeholder
    }

    fn parse_fusion_methods(&mut self) -> Result<Vec<FusionMethod>, TurbulanceError> {
        Ok(Vec::new()) // Placeholder
    }

    fn parse_validation_steps(&mut self) -> Result<Vec<ValidationStep>, TurbulanceError> {
        Ok(Vec::new()) // Placeholder
    }

    fn finish_call(&mut self, expr: Expression) -> Result<Expression, TurbulanceError> {
        let arguments = self.parse_argument_list()?;
        self.consume(Token::RightParen, "Expected ')'")?;
        
        Ok(Expression::FunctionCall {
            function: Box::new(expr),
            arguments,
            uncertainty_propagation: None,
        })
    }

    fn parse_argument_list(&mut self) -> Result<Vec<Expression>, TurbulanceError> {
        let mut arguments = Vec::new();
        
        if !self.check(&Token::RightParen) {
            loop {
                arguments.push(self.parse_expression()?);
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        Ok(arguments)
    }

    fn parse_array_elements(&mut self) -> Result<Vec<Expression>, TurbulanceError> {
        let mut elements = Vec::new();
        
        if !self.check(&Token::RightBracket) {
            loop {
                elements.push(self.parse_expression()?);
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        Ok(elements)
    }

    fn parse_object_literal(&mut self) -> Result<HashMap<String, Expression>, TurbulanceError> {
        let mut object = HashMap::new();
        
        if !self.check(&Token::RightBrace) {
            loop {
                let key = self.parse_string_literal()?;
                self.consume(Token::Colon, "Expected ':'")?;
                let value = self.parse_expression()?;
                object.insert(key, value);
                
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        Ok(object)
    }

    fn parse_string_literal(&mut self) -> Result<String, TurbulanceError> {
        if let Some(token) = self.peek_token() {
            if let Token::String(value) = &token.token {
                let result = value.clone();
                self.advance();
                Ok(result)
            } else {
                Err(TurbulanceError::parse_error(
                    "Expected string literal",
                    Some(token.position.clone()),
                    None,
                    None,
                ))
            }
        } else {
            Err(TurbulanceError::parse_error(
                "Expected string literal",
                None,
                None,
                None,
            ))
        }
    }

    fn parse_float_literal(&mut self) -> Result<f64, TurbulanceError> {
        if let Some(token) = self.peek_token() {
            match &token.token {
                Token::Float(value) => {
                    let result = *value;
                    self.advance();
                    Ok(result)
                }
                Token::Integer(value) => {
                    let result = *value as f64;
                    self.advance();
                    Ok(result)
                }
                _ => Err(TurbulanceError::parse_error(
                    "Expected numeric literal",
                    Some(token.position.clone()),
                    None,
                    None,
                ))
            }
        } else {
            Err(TurbulanceError::parse_error(
                "Expected numeric literal",
                None,
                None,
                None,
            ))
        }
    }

    // Utility methods
    fn peek_token(&self) -> Option<&PositionedToken> {
        self.tokens.get(self.current)
    }

    fn advance(&mut self) -> Option<&PositionedToken> {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous_token()
    }

    fn previous_token(&self) -> Option<&PositionedToken> {
        if self.current > 0 {
            self.tokens.get(self.current - 1)
        } else {
            None
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }

    fn check(&self, token_type: &Token) -> bool {
        if let Some(token) = self.peek_token() {
            std::mem::discriminant(&token.token) == std::mem::discriminant(token_type)
        } else {
            false
        }
    }

    fn match_token(&mut self, token_type: &Token) -> bool {
        if self.check(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn consume(&mut self, token_type: Token, message: &str) -> Result<&PositionedToken, TurbulanceError> {
        if self.check(&token_type) {
            Ok(self.advance().unwrap())
        } else {
            Err(TurbulanceError::parse_error(
                message,
                self.current_position(),
                Some(vec![format!("{:?}", token_type)]),
                self.peek_token().map(|t| format!("{}", t.token)),
            ))
        }
    }

    fn current_position(&self) -> Option<TokenPosition> {
        self.peek_token().map(|t| t.position.clone())
    }

    fn previous_position(&self) -> Option<TokenPosition> {
        self.previous_token().map(|t| t.position.clone())
    }

    fn synchronize(&mut self) {
        self.advance();
        
        while !self.is_at_end() {
            if let Some(token) = self.previous_token() {
                if matches!(token.token, Token::Semicolon) {
                    return;
                }
            }
            
            if let Some(token) = self.peek_token() {
                match token.token {
                    Token::Item | Token::Function | Token::Proposition | Token::Goal
                    | Token::Evidence | Token::Metacognitive | Token::Config => return,
                    _ => {}
                }
            }
            
            self.advance();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::TurbulanceLexer;

    #[test]
    fn test_parse_variable_declaration() {
        let source = "item x = 42";
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        let mut parser = TurbulanceParser::new(tokens);
        
        let item = parser.parse_item().unwrap();
        match item {
            Item::Variable(var_decl) => {
                assert_eq!(var_decl.name.name, "x");
                match var_decl.value {
                    Expression::Literal(Literal::Integer(42)) => {},
                    _ => panic!("Expected integer literal 42"),
                }
            }
            _ => panic!("Expected variable declaration"),
        }
    }

    #[test]
    fn test_parse_function_declaration() {
        let source = "funxn test(): return 42";
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        let mut parser = TurbulanceParser::new(tokens);
        
        let item = parser.parse_item().unwrap();
        match item {
            Item::Function(func_decl) => {
                assert_eq!(func_decl.name.name, "test");
                assert_eq!(func_decl.parameters.len(), 0);
                assert!(!func_decl.is_async);
            }
            _ => panic!("Expected function declaration"),
        }
    }

    #[test]
    fn test_parse_proposition() {
        let source = r#"
            proposition TestProp:
                motion TestMotion("description")
        "#;
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        let mut parser = TurbulanceParser::new(tokens);
        
        let item = parser.parse_item().unwrap();
        match item {
            Item::Proposition(prop_decl) => {
                assert_eq!(prop_decl.name.name, "TestProp");
                assert_eq!(prop_decl.motions.len(), 1);
                assert_eq!(prop_decl.motions[0].name.name, "TestMotion");
                assert_eq!(prop_decl.motions[0].description, "description");
            }
            _ => panic!("Expected proposition declaration"),
        }
    }

    #[test]
    fn test_parse_binary_expression() {
        let source = "x + y * z";
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        let mut parser = TurbulanceParser::new(tokens);
        
        let expr = parser.parse_expression().unwrap();
        match expr {
            Expression::BinaryOp { left, operator, right } => {
                assert!(matches!(operator, BinaryOperator::Add));
                match *left {
                    Expression::Identifier(id) => assert_eq!(id.name, "x"),
                    _ => panic!("Expected identifier x"),
                }
                match *right {
                    Expression::BinaryOp { operator, .. } => {
                        assert!(matches!(operator, BinaryOperator::Multiply));
                    }
                    _ => panic!("Expected multiplication"),
                }
            }
            _ => panic!("Expected binary operation"),
        }
    }

    #[test]
    fn test_parse_probabilistic_expression() {
        let source = "0.95 ± 0.03";
        let tokens = TurbulanceLexer::tokenize(source).unwrap();
        let mut parser = TurbulanceParser::new(tokens);
        
        let expr = parser.parse_expression().unwrap();
        match expr {
            Expression::BinaryOp { left, operator, right } => {
                assert!(matches!(operator, BinaryOperator::PlusMinus));
                match (*left, *right) {
                    (Expression::Literal(Literal::Float(l)), Expression::Literal(Literal::Float(r))) => {
                        assert_eq!(l, 0.95);
                        assert_eq!(r, 0.03);
                    }
                    _ => panic!("Expected float literals"),
                }
            }
            _ => panic!("Expected plus-minus operation"),
        }
    }
} 