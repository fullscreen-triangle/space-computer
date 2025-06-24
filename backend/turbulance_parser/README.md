# Turbulance Parser

A comprehensive Rust-based parser and compiler for the Turbulance probabilistic scientific programming language.

## Overview

The Turbulance parser is designed to handle the complex syntax of probabilistic scientific programming, including:

- **Propositions**: Scientific hypotheses with evidence-based reasoning
- **Motions**: Sub-hypotheses within propositions  
- **Evidence Integration**: Multi-source data fusion with uncertainty quantification
- **Goal Optimization**: Objective function optimization with constraints
- **Metacognitive Analysis**: Self-monitoring and adaptive reasoning
- **Probabilistic Constructs**: Native uncertainty handling and propagation
- **Temporal Analysis**: Time-series processing and pattern detection
- **Verification Systems**: AI understanding validation

## Architecture

### Core Components

1. **Lexer** (`src/lexer.rs`)
   - Tokenizes Turbulance source code using the Logos crate
   - Handles all Turbulance keywords, operators, and literals
   - Maintains position information for error reporting
   - Supports probabilistic operators like `±` and scientific notation

2. **Parser** (`src/parser.rs`)
   - Recursive descent parser converting tokens to AST
   - Handles complex nested structures and probabilistic expressions
   - Error recovery and synchronization
   - Support for all Turbulance language constructs

3. **AST** (`src/ast.rs`)
   - Complete Abstract Syntax Tree definitions
   - Represents all Turbulance language constructs
   - Serializable for intermediate representation storage
   - Rich type system with uncertainty propagation

4. **Type System** (`src/types.rs`)
   - Runtime value representation with uncertainty
   - Probabilistic values and distributions
   - Evidence collections and quality metrics
   - Goal states and optimization tracking
   - Verification results and pattern matching

5. **Error Handling** (`src/error.rs`)
   - Comprehensive error types with position information
   - Diagnostic severity levels and recovery strategies
   - Specialized errors for probabilistic reasoning
   - Error collection and reporting utilities

6. **Engine** (`src/lib.rs`)
   - Main entry point for parsing and compilation
   - Configuration management
   - Execution context handling
   - Integration with Space Computer platform

## Language Features Supported

### Basic Syntax
```turbulance
// Variable declarations
item x = 42
item uncertain_value = 3.14 ± 0.02

// Function definitions
funxn calculate_confidence(data):
    return analyze_uncertainty(data)

// Probabilistic expressions
item result = measurement with_confidence(0.95)
```

### Propositions and Motions
```turbulance
proposition BiomechanicalOptimization:
    context athlete_data = load_elite_athletes()
    
    motion TechniqueEfficiency("Optimal joint angles maximize power transfer")
    motion InjuryPrevention("Movement patterns minimize stress concentrations")
    
    within athlete_data:
        given joint_angle_efficiency() > 0.85 with_confidence(0.8):
            support TechniqueEfficiency with_weight(0.9)
            
        given stress_concentration() < 0.3:
            support InjuryPrevention with_weight(0.85)
```

### Evidence Integration
```turbulance
evidence BiomechanicalData:
    sources:
        - type: "motion_capture"
          reliability: 0.95
          bias: "lighting_conditions"
        - type: "force_plates"
          reliability: 0.98
          bias: "calibration_drift"
    
    collection:
        frequency: "30fps"
        validation_method: "cross_validation"
        quality_threshold: 0.85
    
    processing:
        - name: "noise_reduction"
          operation: "butterworth_filter"
          parameters: {cutoff: 10, order: 4}
        - name: "gap_filling"
          operation: "cubic_spline"
```

### Goal Optimization
```turbulance
goal PerformanceOptimization = Goal.new(
    description: "Maximize athletic performance while minimizing injury risk",
    objectives: [
        maximize(power_output) with_weight(0.4),
        minimize(injury_risk) with_weight(0.6),
        target(technique_score, 0.9) with_weight(0.3)
    ],
    success_threshold: 0.85,
    optimization_algorithm: "multi_objective_genetic",
    personalization_factors: {
        athlete_experience: "elite",
        injury_history: load_history(),
        biomechanical_profile: athlete_profile
    }
)
```

### Metacognitive Analysis
```turbulance
metacognitive QualityAssurance:
    track:
        - name: "confidence_consistency"
          metrics: {variance: confidence_variance(), trend: confidence_trend()}
        - name: "evidence_quality"
          metrics: {completeness: evidence_completeness(), reliability: evidence_reliability()}
    
    evaluate:
        - name: "prediction_accuracy"
          expression: compare_predictions_to_outcomes()
          threshold: 0.8
    
    adapt:
        - condition: confidence_variance() > 0.2
          actions: [increase_evidence_collection(), refine_methodology()]
```

### Uncertainty Handling
```turbulance
// Native uncertainty support
item measurement = 9.81 ± 0.02  // Gaussian uncertainty
item confidence_interval = [9.79, 9.83] with_confidence(0.95)

// Uncertainty propagation
item calculated_result = complex_calculation(measurement) 
    uncertainty_propagation: monte_carlo(samples: 10000)

// Probabilistic reasoning
item reliability = ~Beta(alpha: 8, beta: 2)  // Beta distribution
```

## Usage

### Basic Parsing
```rust
use turbulance_parser::TurbulanceEngine;

let engine = TurbulanceEngine::new();
let source = r#"
    proposition TestProp:
        motion TestMotion("test description")
"#;

match engine.parse(source) {
    Ok(ast) => {
        println!("Parsed successfully: {:#?}", ast);
    }
    Err(e) => {
        eprintln!("Parse error: {}", e);
    }
}
```

### Full Compilation and Execution
```rust
use turbulance_parser::{TurbulanceEngine, EngineConfig};

let config = EngineConfig {
    uncertainty_model: "bayesian_inference".to_string(),
    confidence_threshold: 0.75,
    verification_required: true,
    real_time_analysis: false,
    ..Default::default()
};

let mut engine = TurbulanceEngine::with_config(config);

let source = include_str!("complex_analysis.turb");

match engine.run(source).await {
    Ok(result) => {
        println!("Execution completed successfully");
        println!("Confidence: {:.2}", result.uncertainty_metrics.overall_confidence);
        
        for (prop_id, prop_result) in result.propositions {
            println!("Proposition {}: {:.2} confidence", prop_id, prop_result.confidence);
        }
        
        for recommendation in result.recommendations {
            println!("Recommendation: {}", recommendation.description);
        }
    }
    Err(e) => {
        eprintln!("Execution failed: {}", e);
    }
}
```

## Integration with Space Computer

The Turbulance parser is designed to integrate seamlessly with the Space Computer biomechanical analysis platform:

- **Data Sources**: Direct integration with motion capture, force plates, and other sensors
- **Analysis Pipelines**: Probabilistic processing of biomechanical data
- **Visualization**: Uncertainty-aware rendering of analysis results  
- **Real-time Processing**: Stream processing for live athlete monitoring
- **Expert Knowledge**: Integration of domain expertise through propositions

## Development Status

### Implemented ✅
- Complete lexer with all Turbulance tokens
- Recursive descent parser for core language constructs
- Comprehensive AST definitions
- Rich type system with uncertainty support
- Error handling and diagnostics
- Basic proposition and motion parsing
- Evidence integration framework

### In Progress 🚧
- Complete parser implementation for all constructs
- Compiler backend (AST → executable tasks)
- Runtime executor with probabilistic reasoning
- Goal optimization algorithms
- Metacognitive analysis engine
- Verification system integration

### Planned 📋
- LLVM backend for high-performance execution
- Python interop for scientific libraries
- GPU acceleration for Monte Carlo simulations
- Distributed execution for large-scale analysis
- IDE integration with language server protocol
- Comprehensive test suite and benchmarks

## Building

```bash
cd backend/turbulance_parser
cargo build --release
```

### Running Tests
```bash
cargo test
```

### Benchmarks
```bash
cargo bench
```

## Contributing

The Turbulance parser is part of the Space Computer project. When contributing:

1. Ensure all tests pass
2. Add comprehensive tests for new features
3. Document new language constructs
4. Follow Rust best practices and idioms
5. Consider performance implications for real-time use

## Dependencies

- **logos**: Fast lexical analysis
- **nom**: Parser combinators (alternative parsing)
- **serde**: Serialization for AST and values
- **tokio**: Async runtime for execution
- **nalgebra**: Linear algebra for scientific computing
- **statrs**: Statistical distributions and functions
- **uuid**: Unique identifiers for tracking
- **chrono**: Date and time handling
- **thiserror**: Error handling
- **miette**: Advanced error reporting

## License

Part of the Space Computer project. See main project LICENSE for details. 