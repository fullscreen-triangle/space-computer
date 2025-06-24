# Video Integration with Space Computer Platform

This document demonstrates how to integrate your annotated biomechanical videos with the Space Computer 3D visualization platform.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Space Computer Platform                      │
├─────────────────────┬───────────────────────────────────────────┤
│   Video Reference   │           3D Model Visualization          │
│                     │                                           │
│ ┌─────────────────┐ │ ┌───────────────────────────────────────┐ │
│ │  Annotated      │ │ │  3D Mannequin Model                   │ │
│ │  Video Playback │ │ │  - Joint highlighting                 │ │
│ │                 │ │ │  - Real-time pose sync                │ │
│ │ - Usain Bolt    │ │ │  - Physics simulation                 │ │
│ │ - Drogba        │ │ │                                       │ │
│ │ - Chisora       │ │ │ ┌─────────────────────────────────────┐ │ │
│ │ - etc.          │ │ │ │        AI Chat Interface            │ │ │
│ └─────────────────┘ │ │ │  "What's happening with this knee?" │ │ │
│                     │ │ └─────────────────────────────────────┘ │ │
└─────────────────────┴───────────────────────────────────────────┘
```

## Implementation Guide

### 1. VideoReference Component

The `VideoReference` component plays your annotated videos as reference material:

```tsx
<VideoReference
  videoUrl="/datasources/annotated/usain_bolt_final.mp4"
  athleteName="Usain Bolt"
  sport="Sprint"
  position="left"           // left | right | background | picture-in-picture
  size="half-screen"        // small | medium | large | half-screen
  videoDuration={10.5}
/>
```

### 2. Data Integration

The `BiomechanicalDataLoader` processes your JSON data:

```tsx
// Load athlete data
const athleteData = await dataLoader.loadAthleteData('usain_bolt_final');

// Get frame-synchronized data
const currentPose = dataLoader.getFrameData('usain_bolt_final', frameNumber);
const postureAnalysis = dataLoader.getPostureAnalysis('usain_bolt_final', frameNumber);
```

### 3. Layout Configurations

#### Split Screen Layout (Recommended)
```tsx
<VideoAnalysisComposition 
  athleteId="usain_bolt_final"
  videoPosition="left"
  videoSize="half-screen"
/>
```

#### Picture-in-Picture Layout
```tsx
<VideoAnalysisComposition 
  athleteId="didier_drogba_header"
  videoPosition="picture-in-picture"
  videoSize="medium"
/>
```

#### Background Reference Layout
```tsx
<VideoAnalysisComposition 
  athleteId="derek_chisora_punch"
  videoPosition="background"
  videoSize="large"
/>
```

## Available Athletes

Your datasources include world-class athletes:

| Athlete | Sport | Video File | Model Data |
|---------|-------|------------|------------|
| Usain Bolt | Sprint | `usain_bolt_final.mp4` | `usain_bolt_final.json` |
| Didier Drogba | Football | `didier_drogba_header.mp4` | `didier_drogba_header.json` |
| Derek Chisora | Boxing | `derek_chisora_punch.mp4` | `derek_chisora_punch.json` |
| Jonah Lomu | Rugby | `jonah_lomu_run.mp4` | `jonah_lomu_run.json` |
| Asafa Powell | Sprint | `asafa_powell_race.mp4` | `asafa_powell_race.json` |
| Mahela Jayawardene | Cricket | `mahela_jayawardene_shot.mp4` | `mahela_jayawardene_shot.json` |
| Kevin Pietersen | Cricket | `kevin_pietersen_shot.mp4` | `kevin_pietersen_shot.json` |
| Daniel Sturridge | Football | `daniel_sturridge_dribble.mp4` | `daniel_sturridge_dribble.json` |
| Gareth Bale | Football | `gareth_bale_kick.mp4` | `gareth_bale_kick.json` |
| Jordan Henderson | Football | `jordan_henderson_pass.mp4` | `jordan_henderson_pass.json` |
| Raheem Sterling | Football | `raheem_sterling_sprint.mp4` | `raheem_sterling_sprint.json` |
| Wrestling Analysis | Wrestling | `wrestling_takedown.mp4` | `wrestling_takedown.json` |
| Boxing Analysis | Boxing | `boxing_combo.mp4` | `boxing_combo.json` |

## AI Integration Features

The platform includes click-to-ask AI functionality:

1. **Video Context Awareness**: Click on any part of the video to ask questions
2. **3D Model Interaction**: Click joints in the 3D model for biomechanical insights
3. **Real-time Analysis**: AI understands current frame context
4. **Natural Language**: Ask questions like:
   - "Why is this knee angle important for sprinting?"
   - "How does Bolt's foot strike compare to optimal technique?"
   - "What forces are acting on Drogba's shoulder during this header?"

## Usage Examples

### Basic Setup
```tsx
import { VideoAnalysisComposition } from './src/remotion/VideoAnalysisComposition';

// Render Usain Bolt sprint analysis
<VideoAnalysisComposition 
  athleteId="usain_bolt_final"
  videoPosition="left"
  videoSize="half-screen"
/>
```

### Multi-Sport Comparison
```tsx
// Compare different sports techniques
const athletes = [
  'usain_bolt_final',      // Sprint
  'raheem_sterling_sprint', // Football sprint
  'asafa_powell_race'      // Sprint comparison
];

athletes.map(athleteId => (
  <VideoAnalysisComposition 
    key={athleteId}
    athleteId={athleteId}
    videoPosition="picture-in-picture"
    videoSize="small"
  />
));
```

### Technical Analysis Focus
```tsx
// Deep dive into specific technique
<VideoAnalysisComposition 
  athleteId="derek_chisora_punch"
  videoPosition="background"
  videoSize="large"
  // This creates an overlay where 3D model is prominent
  // with video as reference background
/>
```

## Data Flow

1. **Video Playback**: Remotion syncs video timeline with analysis
2. **Frame Extraction**: Current frame number calculated from video time
3. **Data Lookup**: JSON pose data retrieved for current frame
4. **3D Sync**: Mannequin model updated with pose data
5. **Analysis Update**: Real-time metrics calculated and displayed
6. **AI Context**: Current frame data provides context for AI chat

## Next Steps

1. **Copy Videos**: Move your annotated videos to `space-computer/public/datasources/annotated/`
2. **Copy Data**: Move JSON files to `space-computer/public/datasources/models/`
3. **Test Integration**: Run a simple composition with one athlete
4. **Customize Layout**: Adjust video position and size for your use case
5. **AI Enhancement**: Connect AI backend for conversational analysis

This integration transforms your static analysis data into an interactive, AI-powered biomechanical exploration platform where users can simultaneously watch the real athlete performance and analyze the 3D biomechanical model with intelligent insights. 