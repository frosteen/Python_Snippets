#!/usr/bin/env python3

# You can place regular expressions in regex101.com

import re

text_to_search = """abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
coreyms.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T

romnegrillo@gmail.com
rom.negrillo@gmail.com
rjanegrillo@mycompanyname.domain.com

key=value1
key = value2
key = value with space

Rom Negrillo
"""

def match_pattern(text_to_search, pattern):

  result = re.findall(pattern, text_to_search)

  if result:
    print("Found pattern.") 
    print(result)
  else:
    print("Pattern not found.")
 
def main():
  print(text_to_search)

  print("\n================\n")

  pattern = r"Rom"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"\d"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"\d\d\d"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"\d\d\d-\d\d\d-\d\d\d"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"."
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"\d\d\d.\d\d\d.\d\d\d"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"\w"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"\w+"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"\w+\. \w+"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)


  pattern = r"^a[a-z]+"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)

  pattern = r"[\w\.\-]+@[\w\.\-]+\.com"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)


  pattern = r"key\s*=\s*(.+)"
  print("Finding pattern: {}".format(pattern))
  match_pattern(text_to_search, pattern)




if __name__ == "__main__":
  main()