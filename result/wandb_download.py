import wandb
import os

# ✅ 설정
ENTITY = "2yunsu"        # W&B 팀/사용자 이름
PROJECT = "diral"      # 프로젝트 이름
SAVE_DIR = "./"   # 저장할 폴더

# ✅ 저장 디렉토리 생성
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ W&B API 사용
api = wandb.Api()

# ✅ 프로젝트 내 모든 run 가져오기
runs = api.runs(f"{ENTITY}/{PROJECT}")

# ✅ 각 run에 대해 raw 로그 다운로드
for run in runs:
    run_id = run.id
    run_name = run.name.replace("/", "_")  # 파일명에 사용할 수 있도록 '/' 제거
    print(f"Downloading run: {run_name} ({run_id})")

    # Step별 history 데이터 (raw log)
    history = run.history()

    # CSV 저장
    filename = os.path.join(SAVE_DIR, f"{run_name}_{run_id}.csv")
    history.to_csv(filename, index=False)

print("✅ 모든 run의 raw 데이터가 저장되었습니다.")
