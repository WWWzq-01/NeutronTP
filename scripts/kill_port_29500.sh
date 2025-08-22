#!/bin/bash

# Script to kill processes monitoring port 29500
# Usage: ./kill_port_29500.sh

PORT=29500

echo "Searching for processes using port $PORT..."

# Get PIDs of processes using the port
PIDS=$(lsof -ti :$PORT 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "No processes found using port $PORT"
    exit 0
fi

echo "Found processes using port $PORT:"
lsof -i :$PORT

echo ""
echo "PIDs to kill: $PIDS"
echo "Killing processes..."

# Kill the processes
for PID in $PIDS; do
    echo "Killing process $PID..."
    kill -9 $PID 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Process $PID killed successfully"
    else
        echo "Failed to kill process $PID (may already be dead)"
    fi
done

echo ""
echo "Verifying port $PORT is free..."
REMAINING=$(lsof -ti :$PORT 2>/dev/null)
if [ -z "$REMAINING" ]; then
    echo "Port $PORT is now free"
else
    echo "Warning: Some processes may still be using port $PORT:"
    lsof -i :$PORT
fi