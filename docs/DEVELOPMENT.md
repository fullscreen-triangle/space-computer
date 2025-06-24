# Development Guide

## Project Overview

The Space Computer is a sophisticated 3D biomechanical analysis platform that combines:
- **Remotion** for video rendering and frame-based animations
- **React Three Fiber** for 3D graphics and WebGL rendering
- **GPU.js** for high-performance physics calculations
- **Web Workers** for background processing

## Architecture

### Component Hierarchy
```
MainComposition (Remotion root)
├── ThreeCanvas (3D scene)
│   ├── Lighting
│   ├── Ground plane
│   └── 3D models
├── Analysis Overlays
│   ├── AthleteTracker (SVG overlay)
│   ├── PoseVisualization (SVG skeleton)
│   ├── MotionMetrics (UI panel)
│   ├── BiomechanicalFeedback (UI panel)
│   ├── KinematicsDisplay (UI panel)
│   └── SymmetryAnalysis (UI panel)
└── Controls
    ├── Analysis toggle
    ├── Joint selection
    └── MannequinViewer
```

### Data Flow
1. **Frame-based Updates**: Remotion provides current frame number
2. **Sample Data**: Components use frame to index into sample data
3. **Real-time Rendering**: 60fps updates with physics simulation
4. **User Interaction**: Joint selection and analysis toggles

## Key Components

### MainComposition.tsx
- Root Remotion composition
- Manages global state (highlighted joints, analysis mode)
- Coordinates all analysis components
- Provides sample data for demonstration

### Biomechanical Components
- **AthleteTracker**: Bounding box tracking with confidence scores
- **PoseVisualization**: 2D skeleton overlay on video
- **MotionMetrics**: Performance metrics (speed, stride, etc.)
- **BiomechanicalFeedback**: Joint loads and movement analysis
- **KinematicsDisplay**: Joint angles and velocities
- **SymmetryAnalysis**: Left-right movement symmetry
- **MannequinViewer**: 3D model viewer (simplified for demo)

### Physics System
- **BodyDynamics.ts**: Core physics simulation
- **useBodyModel.ts**: React hook for body model management
- **Types**: Comprehensive TypeScript definitions

## Development Workflow

### Adding New Analysis Components

1. Create component in `src/components/biomechanics/`
2. Define TypeScript interfaces for props
3. Add to `index.ts` exports
4. Integrate into `MainComposition.tsx`

Example:
```tsx
// src/components/biomechanics/NewAnalysis.tsx
interface NewAnalysisProps {
  data: AnalysisData;
  style?: React.CSSProperties;
}

const NewAnalysis: React.FC<NewAnalysisProps> = ({ data, style }) => {
  const frame = useCurrentFrame();
  const currentData = data[frame];
  
  return (
    <div style={style}>
      {/* Analysis visualization */}
    </div>
  );
};

export default NewAnalysis;
```

### Extending Physics Simulation

1. Add new segment types to `BodySegment.ts`
2. Extend `BiomechanicalModel` interface
3. Update physics calculations in `BodyDynamics.ts`
4. Add GPU kernels for performance

### Adding 3D Models

1. Place `.glb` files in `public/models/`
2. Ensure proper bone naming convention
3. Update model URLs in compositions
4. Test with MannequinViewer component

## Performance Considerations

### Frame Rate Optimization
- Use `useCurrentFrame()` for time-based animations
- Minimize re-renders with React.memo
- Leverage GPU.js for heavy calculations
- Use Web Workers for background processing

### Memory Management
- Dispose of Three.js objects properly
- Limit texture sizes and model complexity
- Use object pooling for frequent allocations

### Bundle Size
- Tree-shake unused dependencies
- Optimize 3D models before deployment
- Use dynamic imports for large components

## Testing

### Component Testing
```bash
# Run component tests
npm test

# Test specific component
npm test -- --testNamePattern="MotionMetrics"
```

### Integration Testing
```bash
# Test full composition
npm run test:integration

# Test physics simulation
npm run test:physics
```

### Performance Testing
```bash
# Profile rendering performance
npm run profile

# Test GPU computation
npm run test:gpu
```

## Debugging

### Common Issues

1. **Three.js Import Errors**
   - Use `import * as THREE from 'three'`
   - Check three-stdlib for utilities

2. **Remotion Frame Issues**
   - Ensure `useCurrentFrame()` is called in components
   - Check frame bounds in data indexing

3. **GPU.js Errors**
   - Verify WebGL support
   - Check kernel function syntax

4. **Type Errors**
   - Update TypeScript definitions
   - Use proper interface inheritance

### Debug Tools
- React DevTools for component inspection
- Three.js Inspector for 3D scene debugging
- Remotion Studio for frame-by-frame analysis
- Browser DevTools for performance profiling

## Deployment

### Build Process
```bash
# Development build
npm run dev

# Production build
npm run build

# Render video
npx remotion render MainComposition out.mp4
```

### Environment Variables
```bash
# .env.local
REMOTION_STUDIO_PORT=3000
REMOTION_BROWSER_EXECUTABLE=/path/to/browser
```

### Performance Monitoring
- Monitor frame rates during rendering
- Track memory usage with large datasets
- Profile GPU utilization

## Contributing

### Code Style
- Use TypeScript for all new code
- Follow React functional component patterns
- Document complex physics calculations
- Add JSDoc comments for public APIs

### Pull Request Process
1. Create feature branch from main
2. Add tests for new functionality
3. Update documentation
4. Submit PR with detailed description

### Code Review Checklist
- [ ] TypeScript types are properly defined
- [ ] Components are properly memoized
- [ ] Physics calculations are documented
- [ ] Performance impact is considered
- [ ] Tests are included 