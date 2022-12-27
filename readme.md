# 신규 파일 다운로드
- 50-1 신규 파일 다운로드는 data/update/zip 폴더에 저장
- 이후 `unzip_raw.py` 파일을 실행하면 자동으로 uncombine, combine에 나누어 csv로 저장됨

# 모델 추론
- `python v2x_test.py --model_name=turn(turn, speed)`
- `python v2x_test.py --continue_test "result.txt 파일 경로"` # 이어서 테스트 및 로그 기록