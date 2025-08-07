#!/bin/bash

# Clean Quark Script - No function definitions displayed
# Shows status and defines quark command

QUARK_DIR="/Users/camdouglas/quark"
QUARK_PID_FILE="$QUARK_DIR/logs/quark.pid"
QUARK_READY_FILE="$QUARK_DIR/logs/quark_ready.flag"

# Show status
if [ -f "$QUARK_PID_FILE" ]; then
    PID=$(cat "$QUARK_PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "\033[0;36m🤖 Quark AI System Status\033[0m"
        echo "=================================="
        echo -e "\033[0;32m✅ Quark is running (PID: $PID)\033[0m"
        
        if [ -f "$QUARK_READY_FILE" ]; then
            echo -e "\033[0;32m✅ Ready for user input\033[0m"
            echo -e "\033[0;34m🌐 Web interface: http://localhost:8000\033[0m"
            echo -e "\033[0;34m📊 Metrics: http://localhost:8001\033[0m"
        else
            echo -e "\033[1;33m⏳ Starting up...\033[0m"
        fi
        echo ""
    else
        echo -e "\033[0;36m🤖 Quark AI System Status\033[0m"
        echo "=================================="
        echo -e "\033[0;31m❌ Quark is not running\033[0m"
        echo -e "\033[1;33m💡 Run 'quark start' to start Quark\033[0m"
        echo ""
        
        # Auto-start Quark
        echo -e "\033[1;33m🚀 Auto-starting Quark...\033[0m"
        "$QUARK_DIR/scripts/startup_quark.sh" start > /dev/null 2>&1 &
        sleep 2
        
        # Show updated status
        if [ -f "$QUARK_PID_FILE" ]; then
            PID=$(cat "$QUARK_PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo -e "\033[0;36m🤖 Quark AI System Status\033[0m"
                echo "=================================="
                echo -e "\033[0;32m✅ Quark is running (PID: $PID)\033[0m"
                
                if [ -f "$QUARK_READY_FILE" ]; then
                    echo -e "\033[0;32m✅ Ready for user input\033[0m"
                    echo -e "\033[0;34m🌐 Web interface: http://localhost:8000\033[0m"
                    echo -e "\033[0;34m📊 Metrics: http://localhost:8001\033[0m"
                else
                    echo -e "\033[1;33m⏳ Starting up...\033[0m"
                fi
                echo ""
            fi
        fi
    fi
else
    echo -e "\033[0;36m🤖 Quark AI System Status\033[0m"
    echo "=================================="
    echo -e "\033[0;31m❌ Quark is not running\033[0m"
    echo -e "\033[1;33m💡 Run 'quark start' to start Quark\033[0m"
    echo ""
    
    # Auto-start Quark
    echo -e "\033[1;33m🚀 Auto-starting Quark...\033[0m"
    "$QUARK_DIR/scripts/startup_quark.sh" start > /dev/null 2>&1 &
    sleep 2
    
    # Show updated status
    if [ -f "$QUARK_PID_FILE" ]; then
        PID=$(cat "$QUARK_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "\033[0;36m🤖 Quark AI System Status\033[0m"
            echo "=================================="
            echo -e "\033[0;32m✅ Quark is running (PID: $PID)\033[0m"
            
            if [ -f "$QUARK_READY_FILE" ]; then
                echo -e "\033[0;32m✅ Ready for user input\033[0m"
                echo -e "\033[0;34m🌐 Web interface: http://localhost:8000\033[0m"
                echo -e "\033[0;34m📊 Metrics: http://localhost:8001\033[0m"
            else
                echo -e "\033[1;33m⏳ Starting up...\033[0m"
            fi
            echo ""
        fi
    fi
fi

# Define quark command as a simple alias (no function definition displayed)
alias quark="$QUARK_DIR/scripts/quark" 