import requests
url = "https://0489.oddzial.com/files/301/lotnr2_50525_190959.txt"
try:
    resp = requests.get(url)
    print(resp.text[:1000])
except Exception as e:
    print(e)
