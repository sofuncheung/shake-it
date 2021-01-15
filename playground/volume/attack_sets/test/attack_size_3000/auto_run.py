import os
import time
import subprocess
import sys


def str_in_file_lines(s: str, f: list) -> bool:
    for i in f:
        if s in i:
            return True
    return False

for sample in range(5):
    print('Looking for submitting sample %d...'%(sample+1))
    # check whether there is one running already
    while True:
        print('Checking whether there is task running...')
        os.system("ps -aux | grep shake-it/main > temp_ps_output")
        f = open('temp_ps_output').readlines()
        if not str_in_file_lines("/mnt/zfsusers/sofuncheung/shake-it/playground", f):
            # You know it's not running. So you can submit job.
            print('submitting job...')
            os.system("nohup /mnt/zfsusers/sofuncheung/.venv/pyhessian/bin/python -u "
                "~/shake-it/main.py -p $(pwd) -s %d "
                "--save_checkpoint_on_train_acc > out_%d 2>&1 &" % (sample+1,sample+1))
            print('Rest 5 second... Don\'t rush')
            time.sleep(5)
            break
        else:
            print('Seems like it\'s occupied! Come later young lad...')
            time.sleep(1800)


