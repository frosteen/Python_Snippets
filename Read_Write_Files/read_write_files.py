#!/usr/bin/python3

# Context manager to mange resources
# properly by closing the resources
# such as file afterwards.

# r - read
# w - write
# r+ - read/write
# a - append
# binary

def read_file(file_path):
    """
    This function reads the contents of a file,
    stores each line in the list then iterates over
    the list and prints it.
    """
    with open(file_path, "r") as f:
        
        # Method 1: Read whole contents of the file.
        #f_contents = f.read()

        # Method 2: Read line by line.
        #f_contents = f.readline()

        # Method 3: Loop through the file descriptor.
        # for line in f:
        #     print(line)

        # Read whole contents and place each line in a list.
        f_contents = f.readlines()

        for line in f_contents:
            print(line)

def write_file(file_path, message):
    """
    This function writes message to a file, overwriting its contents.
    """
    with open(file_path, "w") as f:
        f.write(message)

def append_to_file(file_path, message):
    """
    This functions writes message to a file, appending at the end.
    """
    with open(file_path, "a") as f:
        f.write(message)

def read_per_bytesize(file_path, bytesize):
    """
    This function reads the file byte by byte specific by the bytesize.
    One letter is one byte.
    """
    with open("my_file.txt", "r") as f:

        # It is more efficient to read file with specified number of bytes.
        f_contents = f.read(bytesize)

        while len(f_contents) > 0:
            print("Reading {} bytes...".format(bytesize))
            print(f_contents)
            f_contents = f.read(bytesize)      

def main():
    target_file = "my_file.txt"
    read_file(target_file)

    message = "Hello, I'm Rom!"
    write_file(target_file, message)
    read_file(target_file)

    message = "Hi Rom, Welcome!"
    append_to_file(target_file, message)
    read_file(target_file)

    read_per_bytesize(target_file, 512)

if __name__ == "__main__":
    main()