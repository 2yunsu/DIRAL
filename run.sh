#!/bin/bash

# main_test.py를 3번 실행하는 스크립트
# 사용법: ./run_main_test_3times.sh [--test TEST_NUM] [--reward REWARD_NUM]
# 예시: ./run_main_test_3times.sh --test 1 --reward 2
# 작성일: 2025-07-31

# 기본값 설정
TEST_NUM=""
REWARD_NUM=""

# 명령행 인자 파싱
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

# 실행할 명령어 구성
if [[ -n "$TEST_NUM" && -n "$REWARD_NUM" ]]; then
    PYTHON_CMD="python main_test.py --test $TEST_NUM --reward $REWARD_NUM"
    echo "Running with arguments: --test $TEST_NUM --reward $REWARD_NUM"
else
    PYTHON_CMD="python main_test.py"
    echo "Running with default configuration"
fi

echo "Starting main_test.py execution - 3 runs"
echo "========================================"

# 시작 시간 기록
start_time=$(date +%s)

# 3번 반복 실행
for i in {1..3}; do
    echo ""
    echo "Run $i/3 starting at $(date)"
    echo "----------------------------"
    
    # main_test.py 실행 (구성된 명령어 사용)
    $PYTHON_CMD
    
    # 실행 상태 확인
    if [ $? -eq 0 ]; then
        echo "Run $i completed successfully at $(date)"
    else
        echo "Run $i failed with exit code $? at $(date)"
        echo "Stopping execution due to error"
        exit 1
    fi
    
    # 마지막 실행이 아니면 잠시 대기
    if [ $i -lt 3 ]; then
        echo "Waiting 5 seconds before next run..."
        sleep 5
    fi
done

# 종료 시간 기록 및 총 소요 시간 계산
end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "========================================"
echo "All 3 runs completed successfully!"
echo "Total execution time: ${total_time} seconds"
echo "Finished at $(date)"
