import os

os.chdir(".")

for f in os.listdir():
    if f == 'Rename_Folders.py':
        continue
    file_name, file_ext = os.path.splitext(f)
    new_name = ""
    for name in file_name.split(' '):
        filtered_name = name.strip().title()
        new_name = new_name + f"{filtered_name}_"

    new_name = new_name[:-1]+file_ext

    print(new_name)

    # os.rename(f, new_name)



