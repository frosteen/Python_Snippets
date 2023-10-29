#!/usr/bin/python3

import os

def rename_folder_files(target_dir):
    """
    This function renames a structed folder-file pair recursively
    in a certain directory as shown below:

    Folder_Name/File_Name

    turns into

    Folder-Name/File-Name

    I did this to follow the naming conventions I set in some of my
    files.
    """

    for current_dir, dir_list, file_list in os.walk(target_dir):

        if ".git" not in current_dir:
            print(dir_list)
            print(file_list)

            [os.rename(i, i.replace("_", "-")) for i in dir_list]

            file_list_rel_path = [os.path.join(current_dir, i) for i in file_list]

            [os.rename(i, i.replace("_", "-")) for i in file_list_rel_path]

def main():
    rename_folder_files(".")

if __name__ == "__main__":
    main()