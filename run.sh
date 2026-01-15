#!/bin/bash

# Old Money World Model - Run Script

echo "=================================="
echo "Old Money World Model"
echo "=================================="
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Install dependencies if needed
if [ "$1" == "install" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "Installation complete!"
    echo ""
    exit 0
fi

# Run experiments
if [ "$1" == "quickstart" ]; then
    echo "Running quickstart example..."
    python quickstart.py
    
elif [ "$1" == "stable" ]; then
    echo "Running stable attractor experiment..."
    python main.py --experiment stable_attractor --timesteps 500
    
elif [ "$1" == "intervention" ]; then
    echo "Running intervention demo..."
    python main.py --experiment intervention_demo --timesteps 300
    
elif [ "$1" == "multi" ]; then
    echo "Running multi-agent experiment..."
    python main.py --experiment multi_agent --timesteps 400
    
elif [ "$1" == "all" ]; then
    echo "Running all experiments..."
    echo ""
    
    echo "1/3: Stable Attractor Experiment"
    python main.py --experiment stable_attractor --timesteps 500 --save-dir ./results/stable
    echo ""
    
    echo "2/3: Intervention Demo"
    python main.py --experiment intervention_demo --timesteps 300 --save-dir ./results/intervention
    echo ""
    
    echo "3/3: Multi-Agent Experiment"
    python main.py --experiment multi_agent --timesteps 400 --save-dir ./results/multi
    echo ""
    
    echo "All experiments complete! Results saved to ./results/"
    
else
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  install       - Install dependencies"
    echo "  quickstart    - Run quickstart example"
    echo "  stable        - Run stable attractor experiment"
    echo "  intervention  - Run intervention demo"
    echo "  multi         - Run multi-agent experiment"
    echo "  all           - Run all experiments"
    echo ""
    echo "Example: ./run.sh quickstart"
fi

