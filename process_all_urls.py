from flight_agent import process_flight
import logging
import time

# Configure logging
logging.basicConfig(level=logging.WARNING)

urls = [
    # First batch
    "https://0489.oddzial.com/files/301/lotnr2_50525_190959.txt",
    "https://0489.oddzial.com/files/301/lotnr3_120525_190026.txt",
    "https://0489.oddzial.com/files/301/lotnr4_190525_121508.txt",
    "https://0489.oddzial.com/files/301/lotnr5_270525_121136.txt",
    "https://0489.oddzial.com/files/301/lotnr6_310525_194314.txt",
    "https://0489.oddzial.com/files/301/lotnr7_90625_205412.txt",
    "https://0489.oddzial.com/files/301/lotnr8_150625_212100.txt",
    "https://0489.oddzial.com/files/301/lotnr9_220625_202659.txt",
    "https://0489.oddzial.com/files/301/lotnr11_70725_181222.txt",
    "https://0489.oddzial.com/files/301/lotnr1_50524_215640.txt",
    "https://0489.oddzial.com/files/301/lotnr2_120524_203347.txt",
    "https://0489.oddzial.com/files/301/lotnr1_300425_154811.txt",
    # Second batch
    "https://0489.oddzial.com/files/301/lotnr4_260524_194831.txt",
    "https://0489.oddzial.com/files/301/lotnr3_190524_210915.txt",
    "https://0489.oddzial.com/files/301/lotnr5_30624_220832.txt",
    "https://0489.oddzial.com/files/301/lotnr6_90624_194445.txt",
    "https://0489.oddzial.com/files/301/lotnr7_160624_213314.txt",
    "https://0489.oddzial.com/files/301/lotnr8_230624_194634.txt",
    "https://0489.oddzial.com/files/301/lotnr8_230624_194634.txt",
    "https://0489.oddzial.com/files/301/lotnr10_60724_215637.txt"
]

# Remove duplicates
urls = list(dict.fromkeys(urls))

results = []

print(f"Rozpoczynam przetwarzanie {len(urls)} lotów...")

for i, url in enumerate(urls):
    filename = url.split('/')[-1]
    print(f"[{i+1}/{len(urls)}] Przetwarzam: {filename} ... ", end="", flush=True)
    try:
        # Use a new logger for each iteration to avoid clutter
        summary = process_flight(download_url=url, enable_weather=True, force_rebuild=True)
        results.append({
            "url": url,
            "status": "OK",
            "id": summary['flight_id'],
            "info": f"{summary['release_point']} ({summary['flight_date']})"
        })
        print("OK")
    except Exception as e:
        results.append({
            "url": url,
            "status": "ERROR",
            "error": str(e)
        })
        print(f"BŁĄD: {e}")
    time.sleep(0.05)

print("\n--- PODSUMOWANIE ---")
success_count = sum(1 for r in results if r["status"] == "OK")
print(f"Przetworzono poprawnie: {success_count}/{len(urls)}")

for r in results:
    if r["status"] == "OK":
        print(f"[OK] {r['id']} - {r['info']}")
    else:
        print(f"[BŁĄD] {r['url']} -> {r['error']}")
