import os
import torchaudio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# 파일이 있는 디렉토리 경로
directory = ["./dataset/train/clipped_mp3", "./dataset/dev/clipped_mp3", "./dataset/test/clipped_mp3"]
split = ['train', 'dev', 'test']

# 오류가 발생하는 파일을 저장할 리스트
def process_file(file_path):
    """오디오 파일을 로드하고 오류가 발생하면 파일 이름 반환"""
    try:
        waveform, sr = torchaudio.load(file_path, normalize=True, format="mp3")
    except Exception as e:
        return file_path
    return None

for sp, dir in zip(split, directory):
    error_files = []

    # 디렉토리 내 모든 MP3 파일 목록 가져오기
    file_paths = [os.path.join(dir, filename) for filename in os.listdir(dir) if filename.endswith(".mp3")]

    # 멀티쓰레딩 실행
    with ThreadPoolExecutor(max_workers=8) as executor:  # 최대 8개의 쓰레드 사용
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))

    # 오류가 발생한 파일만 필터링
    error_files = [os.path.basename(file_path) for file_path in results if file_path is not None]

    # 결과를 파일로 저장
    os.makedirs("./error_files", exist_ok=True)
    with open(f"./error_files/{sp}_error_files.txt", "w") as f:
        for error_file in error_files:
            f.write(error_file + "\n")

for sp in split:

    # 오류가 발생한 파일 이름 리스트
    error_file_path = f"./error_files/{sp}_error_files.txt"

    # 파일에서 라인별로 읽어와 리스트로 변환
    with open(error_file_path, "r") as f:
        error_files = [line.strip() for line in f]

    # TSV 파일 경로
    tsv_file = f"./dataset/{sp}/covost_v2.en_korean.{sp}.tsv"

    # TSV 파일 읽기
    df = pd.read_csv(tsv_file, sep="\t")

    # 오류가 발생한 파일 이름과 일치하는 행 제거
    df_filtered = df[~df['path'].isin(error_files)]

    # 줄바꿈 문자 (\n)을 공백으로 대체
    df_filtered['sentence'] = df_filtered['sentence'].str.replace("\n", " ")
    df_filtered['translation'] = df_filtered['translation'].str.replace("\n", " ")

    # 결과 저장
    output_file = f"./dataset/{sp}/covost_v2.en_korean.{sp}.filtered.tsv"
    df_filtered.to_csv(output_file, sep="\t", index=False)

    print(f"Filtered TSV saved to {output_file}")