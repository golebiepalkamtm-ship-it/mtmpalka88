import re

line = "                                      Data lotu:  04.05.2025"
pattern = re.compile(r"Data lotu:\s+(?P<value>.+)")
match = pattern.search(line)
if match:
    print(f"Matched: '{match.group('value')}'")
else:
    print("No match")

line_list_name = "                               Lista konkursowa:  nr 2 ODDZIALOWA"
pattern_list_name = re.compile(r"Lista\s+konkursowa:\s+(?P<value>.+)")
match_list = pattern_list_name.search(line_list_name)
if match_list:
    print(f"List Name: '{match_list.group('value')}'")
else:
    print("List Name No match")
