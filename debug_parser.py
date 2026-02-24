from flight_agent import extract_header_from_list, extract_header_data
import requests

url = "https://0489.oddzial.com/files/301/lotnr2_50525_190959.txt"
print(f"Downloading {url}...")
text = requests.get(url).text
lines = text.splitlines()

print(f"Total lines: {len(lines)}")

header_lines = extract_header_from_list(lines)
print(f"Extracted {len(header_lines)} header lines.")

found_date_line = False
for i, line in enumerate(header_lines):
    if "Data lotu" in line:
        print(f"Found 'Data lotu' at line {i}: '{line}'")
        found_date_line = True

if not found_date_line:
    print("WARNING: 'Data lotu' NOT found in header lines!")
    # Print first 20 lines of raw lines to see where it is
    print("First 20 raw lines:")
    for line in lines[:20]:
        print(f"'{line}'")

data = extract_header_data(header_lines)
print("Extracted Data keys:", data.keys())
if "flight_date" in data:
    print(f"flight_date: '{data['flight_date']}'")
else:
    print("flight_date MISSING in data")
