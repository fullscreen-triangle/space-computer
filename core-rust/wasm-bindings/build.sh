#!/bin/bash

# S-Entropy WebAssembly Build Script
# Compiles Rust to WebAssembly with optimization for browser deployment

set -e

echo "🦀 Building S-Entropy Framework WebAssembly..."

# Ensure wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg/
rm -rf target/

# Build for web (ES modules)
echo "🌐 Building for web (ES modules)..."
wasm-pack build \
    --target web \
    --out-dir pkg/web \
    --release \
    --scope space-computer

# Build for bundler (webpack/vite)
echo "📦 Building for bundler..."
wasm-pack build \
    --target bundler \
    --out-dir pkg/bundler \
    --release \
    --scope space-computer

# Build for Node.js
echo "🟢 Building for Node.js..."
wasm-pack build \
    --target nodejs \
    --out-dir pkg/nodejs \
    --release \
    --scope space-computer

# Optimize WASM binary size
echo "⚡ Optimizing WebAssembly binary..."
for dir in pkg/*/; do
    if [ -f "$dir"/*.wasm ]; then
        echo "  Optimizing: $dir"
        # Use wasm-opt if available for further optimization
        if command -v wasm-opt &> /dev/null; then
            for wasm_file in "$dir"/*.wasm; do
                wasm-opt -Oz --enable-simd "$wasm_file" -o "$wasm_file.optimized"
                mv "$wasm_file.optimized" "$wasm_file"
            done
        fi
    fi
done

# Generate TypeScript definitions
echo "📝 Generating TypeScript definitions..."
for dir in pkg/*/; do
    if [ -f "$dir"/package.json ]; then
        # Add TypeScript definitions
        cat > "$dir"/index.d.ts << 'EOF'
/* tslint:disable */
/* eslint-disable */
/**
 * S-Entropy Framework WebAssembly Bindings
 * 
 * Revolutionary zero-computation object detection and S-entropy navigation
 * for browser environments.
 */

export class SEntropyWasm {
  free(): void;
  constructor(st_stella_constant: number);
  static new_golden_ratio(): SEntropyWasm;
  navigate_to_s_coordinate(s_target: number): any;
  compress_gas_field(gas_field_data: any): any;
  is_coherent(): boolean;
  get_performance_stats(): any;
  get_st_stella_constant(): number;
  get_diagnostics(): any;
  batch_navigate_coordinates(s_values: any): any;
  generate_coordinate_manifold(resolution: number): any;
}

export class GasDetectionWasm {
  free(): void;
  constructor(st_stella_constant: number);
  static new_golden_ratio(): GasDetectionWasm;
  detect_objects_zero_computation(baseline_s: number, measured_s: number): any;
  set_baseline_s_value(space_id: string, baseline_s: number): any;
  track_movement_temporal_analysis(s_history: any): any;
  get_detector_status(): any;
  comprehensive_detection_analysis(baseline_s: number, measured_s: number): any;
  reset_statistics(): void;
  static generate_space_id(): string;
}

export class HardwareWasm {
  free(): void;
  constructor();
  detect_browser_hardware_capabilities(): Promise<any>;
  read_gps_differential_s_value(space_id: string): Promise<any>;
  read_camera_led_s_value(space_id: string): Promise<any>;
  read_network_mimo_s_value(space_id: string): any;
  read_oscillatory_harvest_s_value(space_id: string): any;
  read_fused_s_value(space_id: string): Promise<any>;
  get_hardware_status(): any;
}

export class TurbulanceWasm {
  free(): void;
  constructor();
  analyze_s_entropy_probability(s_values: any): any;
  reason_about_s_patterns(s_data: any): any;
  get_turbulance_status(): any;
}

export class PerformanceMonitor {
  free(): void;
  constructor();
  start_operation(): void;
  end_operation(): void;
  get_stats(): any;
}

export class WasmUtils {
  static validate_s_value(s_value: number): boolean;
  static parse_s_values_array(js_array: any): any;
  static log_s_entropy_message(level: string, message: string): void;
  static format_performance_timing(nanoseconds: number): string;
  static generate_uuid(): string;
  static get_current_timestamp(): string;
  static format_memory_improvement(traditional_bytes: number, s_entropy_bytes: number): string;
  static create_error_response(error_type: string, message: string): any;
  static create_success_response(data: any): any;
}

export function get_version_info(): string;
export function validate_module(): boolean;
export function benchmark_zero_computation(): any;
export function test_browser_hardware_integration(): Promise<any>;
export function generate_analysis_report(baseline_s: number, measured_s: number): any;
export function calculate_browser_coordinates(s_value: number): any;
export function handle_s_entropy_error(error_message: string): any;
export function get_high_resolution_time(): number;
export function estimate_js_object_size(js_value: any): number;
export function check_browser_compatibility(): any;
export function format_coordinates_for_js(x: number, y: number, z: number): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
}

export default function init(module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
EOF
    fi
done

echo "✅ S-Entropy WebAssembly build complete!"
echo ""
echo "📁 Build outputs:"
echo "  📦 Web (ES modules): pkg/web/"
echo "  📦 Bundler: pkg/bundler/" 
echo "  📦 Node.js: pkg/nodejs/"
echo ""
echo "🚀 Usage examples:"
echo "  Web: import init, { SEntropyWasm } from './pkg/web/wasm_bindings.js';"
echo "  Bundler: import { SEntropyWasm } from '@space-computer/wasm-bindings';"
echo ""
echo "⚡ Zero-computation object detection ready for browser deployment!"
echo "🌟 Saint Stella-Lorraine's mathematical framework compiled to WebAssembly!"