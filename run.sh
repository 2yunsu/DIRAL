#!/bin/bash

TEST_NUM="1"
REWARD_NUM="2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_NUM="$2"
            shift 2
            ;;
        --reward)
            REWARD_NUM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test TEST_NUM] [--reward REWARD_NUM]"
            echo "Example: $0 --test 1 --reward 2"
            exit 1
            ;;
    esac
done

if [[ -n "$TEST_NUM" && -n "$REWARD_NUM" ]]; then
    PYTHON_CMD="python main_test.py --test $TEST_NUM --reward $REWARD_NUM"
    echo "Running with arguments: --test $TEST_NUM --reward $REWARD_NUM"
else
    PYTHON_CMD="python main_test.py"
    echo "Running with default configuration"
fi

echo "Starting main_test.py execution - 3 runs"
echo "========================================"

start_time=$(date +%s)

for i in {1..3}; do
    echo ""
    echo "Run $i/3 starting at $(date)"
    echo "----------------------------"
    
    $PYTHON_CMD
    
    if [ $? -eq 0 ]; then
        echo "Run $i completed successfully at $(date)"
    else
        echo "Run $i failed with exit code $? at $(date)"
        echo "Stopping execution due to error"
        exit 1
    fi
    
    if [ $i -lt 3 ]; then
        echo "Waiting 5 seconds before next run..."
        sleep 5
    fi
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "========================================"
echo "All 3 runs completed successfully!"
echo "Total execution time: ${total_time} seconds"
echo "Finished at $(date)"
