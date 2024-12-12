import os
import json
import glob
import shutil
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def time_to_milliseconds(time_str):
    h, m, s = map(float, time_str.split(":"))
    return int((h * 3600 + m * 60 + s) * 1000)


def process_audio(args):
    item, audio_folder, output_folder = args
    sn = item["sn"]
    file_name = item["file_name"]
    start_time = item["audio_start"]
    end_time = item["audio_end"]

    source_path = os.path.join(audio_folder, file_name)
    if not os.path.exists(source_path):
        print(f"No source file: {file_name}")
        return None

    try:
        start_ms = time_to_milliseconds(start_time)
        end_ms = time_to_milliseconds(end_time)

        audio = AudioSegment.from_file(source_path)
        trimmed_audio = audio[start_ms:end_ms]

        output_path = os.path.join(output_folder, f"{sn}.mp3")
        trimmed_audio.export(output_path, format="mp3")

        return {
            "audio_root": output_folder,
            "path": f"{sn}.mp3",
            "src_lang": "english",
            "tgt_lang": "korean",
            "sentence": item["source_cleaned"],
            "translation": item["MTPE"],
        }
    except Exception as e:
        print(f"Error for {file_name}: {e}")
        return None


def process_json_files_optimized(json_folder, audio_folder, output_folder, tsv_path):
    os.makedirs(output_folder, exist_ok=True)
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    data_records = []

    for json_file_path in tqdm(json_files):
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file).get("data", [])
        
        # Process each item using threads
        args = [(item, audio_folder, output_folder) for item in data]
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_audio, args))
        
        # Collect valid records
        data_records.extend([r for r in results if r is not None])

    # Save as TSV
    df = pd.DataFrame(data_records)
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"TSV file saved at: {tsv_path}")
    return df


def split_dataset(dev_df, output_audio_folder, test_audio_folder, dev_tsv_path, test_tsv_path):
    os.makedirs(test_audio_folder, exist_ok=True)
    split_index = len(dev_df) // 2
    dev_split = dev_df.iloc[:split_index]
    test_split = dev_df.iloc[split_index:]

    for _, row in test_split.iterrows():
        src_mp3_path = os.path.join(output_audio_folder, row["path"])
        dest_mp3_path = os.path.join(test_audio_folder, row["path"])
        row["audio_root"] = test_audio_folder
        if os.path.exists(src_mp3_path):
            shutil.move(src_mp3_path, dest_mp3_path)
        else:
            print(f"Missing MP3 file: {src_mp3_path}")

    dev_split.to_csv(dev_tsv_path, sep="\t", index=False)
    test_split.to_csv(test_tsv_path, sep="\t", index=False)
    print(f"Dev and Test datasets updated.")
    print(f"Test TSV file saved at: {test_tsv_path}")


def main():
    # Process training data
    process_json_files_optimized(
        json_folder="./data/Training/02.라벨링데이터",
        audio_folder="./data/Training/01.원천데이터",
        output_folder="./dataset/train/clipped_mp3",
        tsv_path="./dataset/train/covost_v2.en_korean.train.tsv",
    )

    # Process validation data and split into dev/test sets
    split_dataset(
        dev_df=process_json_files_optimized(
            json_folder="./data/Validation/02.라벨링데이터",
            audio_folder="./data/Validation/01.원천데이터",
            output_folder="./dataset/dev/clipped_mp3",
            tsv_path="./dataset/dev/covost_v2.en_korean.dev.tsv",
        ),
        output_audio_folder="./dataset/dev/clipped_mp3",
        test_audio_folder="./dataset/test/clipped_mp3",
        dev_tsv_path="./dataset/dev/covost_v2.en_korean.dev.tsv",
        test_tsv_path="./dataset/test/covost_v2.en_korean.test.tsv"
    )


if __name__ == "__main__":
    main()
