#!/bin/bash
# Setup script for Quark AI comprehensive training
# Run with: bash setup_training.sh

echo "🚀 Setting up Quark AI Training Environment"
echo "=========================================="

# Check if we're in the right directory
if [ ! -d "agents" ] || [ ! -d "core" ]; then
    echo "❌ Please run this script from the quark root directory"
    exit 1
fi

echo "✅ Quark directory confirmed"

# Create training directories
echo "📁 Creating training directories..."
mkdir -p training_sessions
mkdir -p training_data
mkdir -p training_logs
mkdir -p training_checkpoints

# Install training requirements
echo "📦 Installing training requirements..."
if command -v pip3 &> /dev/null; then
    pip3 install -r training_requirements.txt
elif command -v pip &> /dev/null; then
    pip install -r training_requirements.txt
else
    echo "❌ pip not found. Please install pip first."
    exit 1
fi

# Set environment variables for training
echo "⚙️  Setting up environment variables..."
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/overnight_comprehensive_training.py
chmod +x scripts/launch_training.py

echo ""
echo "✅ Training environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run: python3 scripts/launch_training.py"
echo "2. Choose your training option"
echo "3. Let it run overnight"
echo ""
echo "📁 Training results will be saved to: training_sessions/"
echo "📝 Logs will be available in each session directory"
echo ""
echo "🌙 Ready for overnight training!"