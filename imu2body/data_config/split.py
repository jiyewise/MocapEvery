from cgi import test
from fileinput import filename
from gc import collect
import random
import numpy as np
import sys, os
from IPython import embed

def collect_filenames(_root_path):
    dirnames = [(f.path, f.name) for f in os.scandir(_root_path) if f.is_dir()]

    filelist = []
    for sub_dir in os.listdir(_root_path):
        sub_dir_path = os.path.join(_root_path, sub_dir)

        if not os.path.isdir(sub_dir_path):
            continue
        if "rub" in sub_dir_path:
            continue
        sub_filelist = []
        for sub_sub_dir in os.listdir(sub_dir_path):
            sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
            if os.path.isdir(sub_sub_dir_path):
                sub_filelist += [os.path.join(sub_dir, sub_sub_dir, filename) for filename in os.listdir(sub_sub_dir_path)]
        
        filelist.append(sub_filelist)
    return filelist


def split_filenames_and_write(filelist):
    train_filenames = []
    test_filenames = []
    validation_filenames = []    

    for sub_filelist in filelist:
        num_files = len(sub_filelist)
        train_end = int(num_files*0.75)
        test_end = int(num_files*0.9)

        random.shuffle(sub_filelist)
        train_filenames += sub_filelist[0:train_end]
        test_filenames += sub_filelist[train_end:test_end]
        validation_filenames += sub_filelist[test_end:]

        print(f"train: {len(train_filenames)} test: {len(test_filenames)} valid: {len(validation_filenames)}")
    
    # write
    fnames_list = ['train', 'test', 'validation']
    for fnames in fnames_list:
        file = open(f"{fnames}_fnames.txt", 'w')
        file.write("\n".join(locals()[f"{fnames}_filenames"]))
        file.close()


if __name__ == "__main__":
    filelist = collect_filenames("../../data/amass/")
    split_filenames_and_write(filelist)