#!/bin/bash

# Space Computer - Elite Athlete Analysis System
# Quick Start Script

echo "🚀 Starting Space Computer System..."
echo "Elite Athlete Biomechanical Analysis Platform"
echo "=============================================="

# Check if datasources exist
if [ ! -d "datasources" ]; then
    echo "❌ Error: datasources folder not found!"
    echo "Please ensure your elite athlete data is in the datasources folder"
    exit 1
fi

# Count data files
MODEL_COUNT=$(find datasources/models -name "*.json" 2>/dev/null | wc -l)
VIDEO_COUNT=$(find datasources/annotated -name "*.mp4" 2>/dev/null | wc -l)

echo "📊 Found $MODEL_COUNT model files and $VIDEO_COUNT video files"

if [ $MODEL_COUNT -eq 0 ]; then
    echo "❌ No model files found in datasources/models/"
    exit 1
fi

# Run integration setup
echo ""
echo "🔧 Running integration setup..."
python scripts/setup_integration.py

if [ $? -ne 0 ]; then
    echo "❌ Integration setup failed"
    exit 1
fi

# Start services
echo ""
echo "🚀 Starting services..."

# Start backend in background
echo "Starting backend server..."
cd backend && python -m uvicorn orchestration.server:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "Starting frontend..."
npm run dev &
FRONTEND_PID=$!

# Print access information
echo ""
echo "✅ Space Computer System Started Successfully!"
echo ""
echo "🌐 Access URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🏆 Elite Athletes Available:"
echo "   • Usain Bolt (Sprint)"
echo "   • Didier Drogba (Football)"
echo "   • Derek Chisora (Boxing)"
echo "   • Jonah Lomu (Rugby)"
echo "   • Asafa Powell (Sprint)"
echo "   • And 8 more world-class athletes..."
echo ""
echo "📝 Quick Actions:"
echo "   1. Open http://localhost:3000"
echo "   2. Select an elite athlete from the dropdown"
echo "   3. Watch synchronized video + 3D visualization"
echo "   4. Get AI-powered biomechanical analysis"
echo ""
echo "⏹️  To stop: Press Ctrl+C"

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down Space Computer System..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ System stopped"
    exit 0
}

# Set trap to handle Ctrl+C
trap cleanup SIGINT

# Wait for processes
wait 