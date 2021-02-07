import os
import sys
import glob
import tqdm


def main():
    in_path = ""
    dir_name = ""
    out_path = ""
    if len(sys.argv) == 4:
        in_path = sys.argv[1]
        dir_name = sys.argv[2]
        out_path = sys.argv[3]
        if out_path == "here":
            out_path = os.getcwd()
    else:
        print("Usage: python3 move_files.py <in_path> <out_path>")

    file_list = glob.glob(in_path+"/"+dir_name+"_*/UNFILTERED/*.bin")

    for file in tqdm.tqdm(file_list, total=len(file_list)):
        if os.path.getsize(file) > 0:
            os.system("mv "+file+" "+out_path)


if __name__ == "__main__":
    main()