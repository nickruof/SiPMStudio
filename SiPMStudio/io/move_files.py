import os
import sys
import glob
import tqdm


def main():
    in_path = ""
    out_path = ""
    if len(sys.argv) == 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
    else:
        print("Usage: python3 move_files.py <in_path> <out_path>")

    file_list = glob.glob(in_path+"/run_*/Unfiltered/*.bin")

    for file in tqdm.tqdm(file_list, total=len(file_list)):
        if os.path.getsize(file) > 0:
            os.system("cp "+file+" "+out_path)


if __name__ == "__main__":
    main()