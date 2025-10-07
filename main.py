import subprocess


def start_training():
    command = "nohup python train.py & echo $! > pid.txt"
    subprocess.run(command, shell=True)
    print("Training started in background. PID stored in pid.txt.")


def stop_training():
    subprocess.run(["pkill", "-f", "train.py"])


if __name__ == '__main__':
    start_training()