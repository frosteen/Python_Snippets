#!/usr/bin/python3

import subprocess


def install():
    """
    This functions executes the command in the commands list.
    It just runs sudo install for all the commonly software
    for Machine Learning.
    """

    message = "This script is used to update and install packages that I normally used in Machine Learning with Python.\n"
    message += "Tested in Ubuntu 20.04.1"

    print(message)
    input("Press enter to cotinue.")

    commands = [
        "pip3 install numpy",
        "pip3 install pillow",
        "pip3 install --upgrade scikit-learn",
        "pip3 install tensorflow",  # or tensorflow-gpu
        # "pip3 install keras",
        "pip3 install opencv-contrib-python",
        "pip3 install imutils",
    ]

    failed_commands = []

    for command in commands:
        print(command)
        result = subprocess.run([command], shell=True, capture_output=True)

        print(result)

        if result.returncode != 0:
            failed_commands.append(command)

    if len(failed_commands):

        print("The following commands failed.")

        for command in failed_commands:
            print(command)

    else:
        print("All commands successfully executed")


def main():
    install()


if __name__ == "__main__":
    main()
