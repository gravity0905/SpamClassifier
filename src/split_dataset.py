import os
import sys
import glob
import random

def split_dataset(source, dest, split=0.3):
    list_files = glob.glob(source) 
    list_files = random.sample(list_files, round(split*len(list_files)))
    for path_name in list_files:
            os.rename(path_name, os.path.join(dest, os.path.basename(path_name)))
    return 0


def split_dataset_default(split=0.3):
    path_bare = r".\lingspam_public\bare"
    path_proc = r".\lingspam_public\lemm_stop"
    paths = [(r'ham', r'ham_test'),
             (r'spam', r'spam_test')]
    for path1, path2 in paths:
        list_files = glob.glob(os.path.join(os.path.join(path_bare, path1), "*")) 
        list_files = random.sample(list_files, round(split*len(list_files)))
        for path_name in list_files:
            base = os.path.basename(path_name)
            os.rename(os.path.join(os.path.join(path_bare, path1), base), os.path.join(os.path.join(path_bare, path2), base))
            os.rename(os.path.join(os.path.join(path_proc, path1), base), os.path.join(os.path.join(path_proc, path2), base))
    return 0

def main():
    if len(sys.argv) != 3:
        # print('usage: ./split_dataset.py source_path destination_path')
        split_dataset_default()
        sys.exit(1)
    split_dataset(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()