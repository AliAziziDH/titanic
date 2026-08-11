import subprocess

def submit():
    subprocess.run(["kaggle", "competitions", "submit", "-c", "titanic", "-f", "submissions/submission_stacking.csv", "-m", "Fix stacking probabilities format"])

if __name__ == "__main__":
    submit()
