import subprocess
import time

def submit():
    print("Submitting to Kaggle...")
    subprocess.run([
        "kaggle", "competitions", "submit", "-c", "titanic",
        "-f", "submissions/submission_stacking.csv",
        "-m", "Fix stacking probabilities format"
    ])

    print("Waiting for evaluation...")
    time.sleep(10) # Give Kaggle some time to evaluate

    print("Polling API for submissions...")
    subprocess.run(["kaggle", "competitions", "submissions", "-c", "titanic"])

if __name__ == "__main__":
    submit()
