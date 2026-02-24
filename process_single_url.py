from flight_agent import process_flight
from pathlib import Path
import logging
import os

import sys

# Configure logging to show info
logging.basicConfig(level=logging.INFO)

if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    url = "https://0489.oddzial.com/files/301/lotnr2_50525_190959.txt"

tmp_path = Path("temp_lot_debug.txt")

print(f"Rozpoczynam przetwarzanie: {url}")

try:
    summary = process_flight(list_path=tmp_path, download_url=url)
    print("\n--- SUKCES ---")
    print(f"ID Lotu: {summary['flight_id']}")
    print(f"Miejscowość: {summary['release_point']}")
    print(f"Data: {summary.get('flight_date', 'N/A')}")
    print(f"Raport zapisany w: {summary.get('report_path', 'N/A')}")
except Exception as e:
    print(f"\n--- BŁĄD ---")
    print(e)
finally:
    if tmp_path.exists():
        os.remove(tmp_path)
