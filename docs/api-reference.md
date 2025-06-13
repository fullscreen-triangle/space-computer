---
layout: default
title: "API Reference"
description: "Complete API documentation for Space Computer integration"
show_toc: true
show_navigation: true
---

# API Reference

The Space Computer API provides comprehensive access to biomechanical analysis, AI processing, and 3D pose manipulation capabilities. Build powerful integrations and custom applications with our RESTful API.

## 🚀 **Getting Started**

### **Base URL**
```
https://api.space-computer.ai/v1
```

### **Authentication**
All API requests require authentication using API keys:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.space-computer.ai/v1/analysis
```

### **Get Your API Key**
1. Sign up at [space-computer.ai](https://space-computer.ai)
2. Navigate to Settings → API Keys
3. Generate a new API key
4. Store securely (keys cannot be recovered)

## 📋 **Rate Limits**

| Plan | Requests/Hour | Video Analysis/Day | 3D Manipulations/Hour |
|------|---------------|-------------------|----------------------|
| Free | 100 | 5 | 50 |
| Pro | 1,000 | 100 | 500 |
| Enterprise | 10,000 | Unlimited | Unlimited |

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## 🎬 **Video Analysis API**

### **Upload Video for Analysis**

**Endpoint**: `POST /analysis/upload`

Upload a video file for biomechanical analysis.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "video=@/path/to/video.mp4" \
  -F "analysis_type=full" \
  -F "sport=tennis" \
  https://api.space-computer.ai/v1/analysis/upload
```

**Parameters**:
- `video` (file, required): Video file (MP4, MOV, AVI)
- `analysis_type` (string): `basic`, `full`, `professional`
- `sport` (string): Sport type for specialized analysis
- `privacy` (string): `public`, `private`, `temporary`

**Response**:
```json
{
  "analysis_id": "ana_1234567890",
  "status": "processing",
  "estimated_completion": "2024-01-15T10:30:00Z",
  "video_info": {
    "duration": 15.5,
    "fps": 30,
    "resolution": "1920x1080",
    "format": "mp4"
  }
}
```

### **Check Analysis Status**

**Endpoint**: `GET /analysis/{analysis_id}`

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.space-computer.ai/v1/analysis/ana_1234567890
```

**Response**:
```json
{
  "analysis_id": "ana_1234567890",
  "status": "completed",
  "progress": 100,
  "created_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:02:30Z",
  "results": {
    "pose_data_url": "https://api.space-computer.ai/v1/analysis/ana_1234567890/poses",
    "metrics_url": "https://api.space-computer.ai/v1/analysis/ana_1234567890/metrics",
    "visualization_url": "https://app.space-computer.ai/analysis/ana_1234567890"
  }
}
```

**Status Values**:
- `queued`: Waiting for processing
- `processing`: Analysis in progress
- `completed`: Analysis finished successfully
- `failed`: Analysis failed (see error details)

### **Get Analysis Results**

**Endpoint**: `GET /analysis/{analysis_id}/results`

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.space-computer.ai/v1/analysis/ana_1234567890/results
```

**Response**:
```json
{
  "analysis_id": "ana_1234567890",
  "metadata": {
    "video_duration": 15.5,
    "total_frames": 465,
    "detected_subjects": 1,
    "confidence_score": 0.94
  },
  "pose_data": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "poses": [
        {
          "subject_id": 1,
          "keypoints_2d": {
            "nose": {"x": 960, "y": 340, "confidence": 0.98},
            "left_eye": {"x": 950, "y": 330, "confidence": 0.97},
            "right_eye": {"x": 970, "y": 330, "confidence": 0.96}
            // ... all 17 keypoints
          },
          "keypoints_3d": {
            "nose": {"x": 0.0, "y": 1.65, "z": 0.1, "confidence": 0.94},
            "left_eye": {"x": -0.03, "y": 1.67, "z": 0.12, "confidence": 0.93}
            // ... all 17 keypoints in 3D space
          }
        }
      ]
    }
    // ... all frames
  ],
  "biomechanical_metrics": {
    "kinematics": {
      "average_velocity": 2.3,
      "peak_velocity": 4.1,
      "acceleration_profile": [0.2, 0.8, 1.2, 0.9, 0.3],
      "joint_angles": {
        "left_knee": {"min": 45, "max": 165, "average": 105},
        "right_knee": {"min": 43, "max": 167, "average": 107}
      }
    },
    "quality_scores": {
      "technique": 0.87,
      "symmetry": 0.92,
      "smoothness": 0.85,
      "coordination": 0.89
    }
  }
}
```

## 🤖 **AI Chat API**

### **Ask Question About Analysis**

**Endpoint**: `POST /ai/query`

Ask natural language questions about biomechanical analysis.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "ana_1234567890",
    "question": "What makes this tennis serve effective?",
    "context": {
      "frame_number": 45,
      "selected_joints": ["right_shoulder", "right_elbow"],
      "user_expertise": "beginner"
    }
  }' \
  https://api.space-computer.ai/v1/ai/query
```

**Response**:
```json
{
  "query_id": "qry_0987654321",
  "question": "What makes this tennis serve effective?",
  "answer": "This serve demonstrates excellent biomechanical efficiency with three key factors: 1) Optimal kinetic chain activation - the shoulder rotation generates 847 Nm of torque, 2) Perfect timing - the elbow extension occurs 0.15 seconds after shoulder rotation for maximum energy transfer, 3) Ball contact point - 2.4m above ground provides ideal trajectory angle.",
  "confidence": 0.94,
  "sources": [
    "biomechanical_analysis",
    "professional_technique_database",
    "physics_simulation"
  ],
  "context_used": {
    "frame_data": true,
    "joint_selection": true,
    "sport_specific_knowledge": true
  },
  "follow_up_suggestions": [
    "How can I improve my shoulder rotation?",
    "Show me the force vectors during ball contact",
    "Compare this to Federer's serve technique"
  ]
}
```

### **Get Conversation History**

**Endpoint**: `GET /ai/conversations/{analysis_id}`

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.space-computer.ai/v1/ai/conversations/ana_1234567890
```

## 🎮 **3D Pose Manipulation API**

### **Create Pose Session**

**Endpoint**: `POST /poses/sessions`

Create a new pose manipulation session.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "ana_1234567890",
    "frame_number": 45,
    "manipulation_type": "biomechanical_optimization"
  }' \
  https://api.space-computer.ai/v1/poses/sessions
```

**Response**:
```json
{
  "session_id": "pos_session_abc123",
  "analysis_id": "ana_1234567890",
  "initial_pose": {
    "keypoints_3d": {
      "nose": {"x": 0.0, "y": 1.65, "z": 0.1},
      "left_shoulder": {"x": -0.2, "y": 1.4, "z": 0.0}
      // ... all keypoints
    }
  },
  "constraints": {
    "biomechanical_limits": true,
    "physics_simulation": true,
    "joint_range_limits": true
  }
}
```

### **Manipulate Joint Position**

**Endpoint**: `POST /poses/sessions/{session_id}/manipulate`

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "joint": "right_shoulder",
    "new_position": {"x": -0.15, "y": 1.45, "z": 0.05},
    "apply_constraints": true
  }' \
  https://api.space-computer.ai/v1/poses/sessions/pos_session_abc123/manipulate
```

**Response**:
```json
{
  "session_id": "pos_session_abc123",
  "manipulation_id": "man_xyz789",
  "updated_pose": {
    "keypoints_3d": {
      // Updated pose with new joint position
    }
  },
  "physics_validation": {
    "is_valid": true,
    "constraint_violations": [],
    "stability_score": 0.89
  },
  "biomechanical_changes": {
    "joint_angles_affected": ["right_elbow", "right_wrist"],
    "force_redistribution": {
      "right_shoulder": {"change": "+12N", "direction": "upward"}
    }
  }
}
```

### **Optimize Pose**

**Endpoint**: `POST /poses/sessions/{session_id}/optimize`

Use AI to optimize the pose for a specific objective.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "maximize_power_output",
    "constraints": ["maintain_balance", "biomechanical_limits"],
    "preserve_joints": ["left_ankle", "right_ankle"]
  }' \
  https://api.space-computer.ai/v1/poses/sessions/pos_session_abc123/optimize
```

## 🏃 **Elite Athlete Data API**

### **List Available Athletes**

**Endpoint**: `GET /athletes`

Get a list of available elite athletes and their techniques.

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.space-computer.ai/v1/athletes
```

**Query Parameters**:
- `sport` (string): Filter by sport type
- `technique` (string): Filter by specific technique
- `skill_level` (string): Filter by skill level
- `page` (integer): Page number for pagination
- `limit` (integer): Number of results per page

**Response**:
```json
{
  "athletes": [
    {
      "id": "ath_1234567890",
      "name": "John Smith",
      "sport": "tennis",
      "techniques": ["serve", "forehand", "backhand"],
      "skill_level": "professional",
      "metadata": {
        "height": 188,
        "weight": 85,
        "experience": "15 years",
        "achievements": ["Grand Slam Winner", "World #1"]
      }
    }
  ],
  "pagination": {
    "total": 150,
    "page": 1,
    "limit": 10,
    "pages": 15
  }
}
```

### **Get Athlete Data**

**Endpoint**: `GET /athletes/{athlete_id}`

Get detailed biomechanical data for a specific athlete.

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.space-computer.ai/v1/athletes/ath_1234567890
```

**Response**:
```json
{
  "athlete_id": "ath_1234567890",
  "name": "John Smith",
  "sport": "tennis",
  "techniques": {
    "serve": {
      "data_url": "https://api.space-computer.ai/v1/athletes/ath_1234567890/techniques/serve",
      "metrics": {
        "ball_speed": 220,
        "spin_rate": 2500,
        "accuracy": 0.85
      },
      "biomechanical_data": {
        "kinematics": {
          "joint_angles": {
            "shoulder": {"min": 0, "max": 180, "average": 90},
            "elbow": {"min": 0, "max": 150, "average": 75}
          },
          "velocities": {
            "shoulder": {"peak": 1200, "average": 800},
            "elbow": {"peak": 900, "average": 600}
          }
        },
        "kinetics": {
          "forces": {
            "ground_reaction": {"peak": 2500, "average": 1800},
            "joint_moments": {"shoulder": 120, "elbow": 80}
          }
        }
      }
    }
  }
}
```

### **Compare with Athlete**

**Endpoint**: `POST /analysis/compare`

Compare user's technique with elite athlete data.

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "ana_1234567890",
    "athlete_id": "ath_1234567890",
    "technique": "serve",
    "comparison_mode": "side-by-side"
  }' \
  https://api.space-computer.ai/v1/analysis/compare
```

**Response**:
```json
{
  "comparison_id": "cmp_1234567890",
  "analysis_id": "ana_1234567890",
  "athlete_id": "ath_1234567890",
  "technique": "serve",
  "comparison_mode": "side-by-side",
  "results": {
    "differences": {
      "shoulder_rotation": {
        "user": 85,
        "athlete": 95,
        "difference": -10,
        "significance": "high"
      },
      "elbow_extension": {
        "user": 140,
        "athlete": 150,
        "difference": -10,
        "significance": "medium"
      }
    },
    "performance_metrics": {
      "ball_speed": {
        "user": 180,
        "athlete": 220,
        "difference": -40,
        "improvement_potential": "high"
      }
    },
    "recommendations": [
      {
        "aspect": "shoulder_rotation",
        "suggestion": "Increase shoulder rotation by 10 degrees",
        "impact": "Expected 15% increase in ball speed"
      }
    ]
  }
}
```

### **Get Comparison Visualization**

**Endpoint**: `GET /analysis/compare/{comparison_id}/visualization`

Get visualization data for technique comparison.

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.space-computer.ai/v1/analysis/compare/cmp_1234567890/visualization
```

**Response**:
```json
{
  "comparison_id": "cmp_1234567890",
  "visualization_data": {
    "side_by_side": {
      "user_url": "https://api.space-computer.ai/v1/visualizations/user_serve",
      "athlete_url": "https://api.space-computer.ai/v1/visualizations/athlete_serve"
    },
    "overlay": {
      "url": "https://api.space-computer.ai/v1/visualizations/overlay_serve"
    },
    "metrics": {
      "url": "https://api.space-computer.ai/v1/visualizations/metrics_serve"
    }
  }
}
```

## 📊 **Analytics API**

### **Get Performance Metrics**

**Endpoint**: `GET /analytics/metrics/{analysis_id}`

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.space-computer.ai/v1/analytics/metrics/ana_1234567890
```

**Response**:
```json
{
  "analysis_id": "ana_1234567890",
  "overall_metrics": {
    "technique_score": 87,
    "efficiency_rating": 0.84,
    "injury_risk_score": 0.23,
    "performance_potential": 0.91
  },
  "detailed_analysis": {
    "kinematics": {
      "velocity_profile": {
        "max_velocity": 4.2,
        "average_velocity": 2.1,
        "velocity_consistency": 0.78
      },
      "acceleration_analysis": {
        "peak_acceleration": 8.5,
        "acceleration_smoothness": 0.82
      }
    },
    "joint_analysis": {
      "range_of_motion": {
        "left_knee": {"rom": 120, "optimal_range": "110-130", "efficiency": 0.88},
        "right_knee": {"rom": 118, "optimal_range": "110-130", "efficiency": 0.86}
      },
      "load_distribution": {
        "left_knee": {"peak_force": 890, "average_force": 450, "safety_margin": 0.72},
        "right_knee": {"peak_force": 920, "average_force": 470, "safety_margin": 0.69}
      }
    }
  },
  "recommendations": [
    {
      "category": "technique",
      "priority": "high",
      "description": "Increase knee flexion during landing phase",
      "expected_improvement": "15% reduction in joint stress"
    }
  ]
}
```

### **Compare with Professional**

**Endpoint**: `POST /analytics/compare`

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_analysis_id": "ana_1234567890",
    "professional_id": "pro_federer_serve_2021",
    "comparison_type": "technique_analysis"
  }' \
  https://api.space-computer.ai/v1/analytics/compare
```

## 🔐 **Webhooks**

### **Setup Webhooks**

**Endpoint**: `POST /webhooks`

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/webhooks/space-computer",
    "events": ["analysis.completed", "analysis.failed"],
    "secret": "your_webhook_secret"
  }' \
  https://api.space-computer.ai/v1/webhooks
```

### **Webhook Events**

**Analysis Completed**:
```json
{
  "event": "analysis.completed",
  "timestamp": "2024-01-15T10:02:30Z",
  "data": {
    "analysis_id": "ana_1234567890",
    "status": "completed",
    "processing_time": 150,
    "confidence_score": 0.94
  }
}
```

## 📚 **SDKs and Libraries**

### **JavaScript/TypeScript**
```bash
npm install @space-computer/sdk
```

```typescript
import { SpaceComputerAPI } from '@space-computer/sdk';

const api = new SpaceComputerAPI('your-api-key');

// Upload and analyze video
const analysis = await api.uploadVideo('./tennis-serve.mp4', {
  sport: 'tennis',
  analysisType: 'full'
});

// Ask AI questions
const response = await api.askQuestion(analysis.id, 
  "How can I improve this serve?", 
  { selectedJoints: ['right_shoulder'] }
);
```

### **Python**
```bash
pip install space-computer-sdk
```

```python
from space_computer import SpaceComputerAPI

api = SpaceComputerAPI('your-api-key')

# Upload and analyze video
analysis = api.upload_video('tennis-serve.mp4', 
                          sport='tennis', 
                          analysis_type='full')

# Ask AI questions
response = api.ask_question(analysis.id, 
                          "How can I improve this serve?",
                          selected_joints=['right_shoulder'])
```

### **cURL Examples Collection**
Complete Postman/Insomnia collection available at:
`https://api.space-computer.ai/v1/docs/collection.json`

## ⚠️ **Error Handling**

### **HTTP Status Codes**
- `200`: Success
- `201`: Created successfully
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `403`: Forbidden (insufficient permissions)
- `404`: Resource not found
- `429`: Rate limit exceeded
- `500`: Internal server error

### **Error Response Format**
```json
{
  "error": {
    "code": "INVALID_VIDEO_FORMAT",
    "message": "Video format not supported. Please use MP4, MOV, or AVI.",
    "details": {
      "supported_formats": ["mp4", "mov", "avi"],
      "received_format": "wmv"
    },
    "request_id": "req_abc123xyz"
  }
}
```

### **Common Error Codes**
- `INVALID_API_KEY`: API key is invalid or expired
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INVALID_VIDEO_FORMAT`: Unsupported video format
- `VIDEO_TOO_LARGE`: Video file exceeds size limit
- `ANALYSIS_NOT_FOUND`: Analysis ID doesn't exist
- `INSUFFICIENT_CREDITS`: Account credits exhausted

## 🔧 **Testing**

### **Sandbox Environment**
Test API endpoints without affecting your quota:
```
https://sandbox-api.space-computer.ai/v1
```

### **Test Video Samples**
Download sample videos for testing:
- [Tennis Serve (5MB)](https://samples.space-computer.ai/tennis-serve.mp4)
- [Running Gait (8MB)](https://samples.space-computer.ai/running-gait.mp4)
- [Basketball Shot (3MB)](https://samples.space-computer.ai/basketball-shot.mp4)

## 📞 **Support**

### **API Support**
- **Email**: api-support@space-computer.ai
- **Documentation**: [docs.space-computer.ai](https://docs.space-computer.ai)
- **Status Page**: [status.space-computer.ai](https://status.space-computer.ai)
- **GitHub Issues**: [github.com/space-computer/api-issues](https://github.com/space-computer/api-issues)

### **Rate Limit Increases**
For higher rate limits, contact enterprise@space-computer.ai with:
- Expected usage patterns
- Integration details
- Business requirements

Start building amazing biomechanical analysis applications today! 🚀 