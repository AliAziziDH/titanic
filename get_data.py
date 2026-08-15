import kagglehub
import shutil
import os

try:
    path = kagglehub.competition_download('titanic')
    os.makedirs('data/raw', exist_ok=True)
    shutil.copy(f"{path}/train.csv", "data/raw/train.csv")
    shutil.copy(f"{path}/test.csv", "data/raw/test.csv")
    shutil.copy(f"{path}/gender_submission.csv", "data/raw/gender_submission.csv")
    print("Data downloaded using kagglehub")
except Exception as e:
    print(f"Error: {e}")
