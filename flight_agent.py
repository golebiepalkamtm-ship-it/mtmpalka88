import argparse
import csv
import math
import datetime as dt
import json
import logging
import re
import time
import unicodedata
import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import openmeteo_requests
import requests
import requests_cache
import pandas as pd
from retry_requests import retry
from requests import Session
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util import Retry

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=2.0)
openmeteo = openmeteo_requests.Client(session=retry_session)

_OPENMETEO_DAILY_LIMIT_REACHED = False
_OPENMETEO_DAILY_LIMIT_LOGGED = False

DATA_ROOT = Path("data")
LOTS_ROOT = DATA_ROOT / "lots"
DB_ROOT = DATA_ROOT / "db"
FLIGHTS_CSV = DB_ROOT / "flights.csv"
FANCIERS_CSV = DB_ROOT / "fanciers.csv"
RESULTS_CSV = DB_ROOT / "results.csv"
ARRIVALS_CSV = DB_ROOT / "arrivals.csv"
SECTIONS_CSV = DB_ROOT / "sections.csv"

FLIGHTS_FIELDNAMES = [
    "flight_id",
    "season",
    "list_name",
    "flight_date",
    "release_time",
    "release_point",
    "release_lat",
    "release_lon",
    "fanciers",
    "birds_sent",
    "birds_returned",
    "avg_distance_km",
    "lot_dir",
]

ARRIVALS_FIELDNAMES = [
    "flight_id",
    "lp",
    "fancier_id",
    "name",
    "ring",
    "category",
    "typed",
    "arrival_time",
    "speed_mpm",
    "distance_km",
]

FANCIERS_FIELDNAMES = [
    "fancier_id",
    "name",
    "section",
    "lat",
    "lon",
    "default_birds_sent",
]

RESULTS_FIELDNAMES = [
    "flight_id",
    "fancier_id",
    "name",
    "section",
    "birds_sent",
    "birds_classified",
    "loss_min_pct",
    "avg_speed",
    "best_speed",
    "course_deg",
    "wind_label",
    "wind_proj_m_s",
    "wind_proj_10m",
    "turbulence_index_10m",
    "wind_effective_10m_kmh",
    "wind_proj_80m",
    "wind_proj_100m",
    "wind_proj_120m",
    "wind_proj_180m",
    "temperature_C",
    "pressure_hPa",
    "precipitation_mm",
    "humidity_pct",
    "cloud_cover_pct",
    "visibility_m",
    "rain_mm",
    "snowfall_mm",
    "showers_mm",
    "surface_pressure_hPa",
    "shortwave_radiation",
    "uv_index",
]

SECTIONS_FIELDNAMES = [
    "flight_id",
    "section",
    "fanciers",
    "sent",
    "returned",
    "percent_returned",
    "percent_birds",
    "loss_pct",
]

HEADER_PATTERNS = {
    "release_point": re.compile(r"Miejsce\s+wypuszczenia:\s+(?P<value>.+)"),
    "release_coords": re.compile(r"Wspolrzedne.*geog.*:\s*(?P<value>.+)"),
    "release_time": re.compile(r"Godzina\s+wypuszczenia:\s+(?P<value>.+)"),
    "flight_date": re.compile(r"Data lotu:\s+(?P<value>.+)"),
    "season": re.compile(r"Sezon:\s+(?P<value>.+)"),
    "list_name": re.compile(r"Lista\s+konkursowa:\s+(?P<value>.+)"),
    "fanciers": re.compile(r"Lic\w+ hodowcow:\s+(?P<value>\d+)"),
    "birds_sent": re.compile(r"Lic\w+ wyslanych golebi:\s+(?P<value>\d+)"),
    "birds_returned": re.compile(r"Lic\w+ golebi, ktore wrocily:\s+(?P<value>\d+)"),
    "top_speed": re.compile(r"Predkosc najszybszego golebia:\s+(?P<value>[\d.,]+)"),
    "avg_distance": re.compile(r"Srednia odleglosc z lotu:\s+(?P<value>[\d.,]+)"),
}

ARRIVAL_PATTERN = re.compile(
    r"^\s*(?P<lp>\d+)\s+"
    r"(?P<name>[A-Za-zĄąĆćĘęŁłŃńÓóŚśŻżŹź0-9\.\-_ ']+?)\s{2,}"
    r"(?P<section>\d+)\s+"
    r"(?P<wkm>[0-9<>+\- ]+)\s+"
    r"(?:(?P<category>(?![A-Z]{2}-)[A-Z]{1,4})\s+)?"
    r"(?P<ring>[A-Z]{2}-[A-Z0-9-]+)\s+"
    r"(?P<typed>[0-9]*)\s*"
    r"(?P<arrival>\d-\d{2}:\d{2}:\d{2})\s+"
    r"(?P<speed>[0-9]+\.[0-9]+)\s+"
    r"(?P<coef>[0-9]+\.[0-9]+)\s+"
    r"(?P<pkt_oddz2>[0-9]+\.[0-9]+)\s+"
    r"(?P<pkt_gmp>[0-9]+\.[0-9]+)\s+"
    r"(?P<pkt_oddz>[0-9]+\.[0-9]+)\s+"
    r"(?P<distance>[0-9]+\.[0-9]+)\s*$",
    re.IGNORECASE,
)

EARTH_RADIUS_KM = 6371.0

DEFAULT_SECTION_REFERENCE = {
    "S1": {"name": "Lubań", "lat": 51.1202, "lon": 15.2875},
    "S2": {"name": "Grudza", "lat": 50.9716, "lon": 15.4853},
    "S3": {"name": "Chmielen", "lat": 51.0004, "lon": 15.5595},
    "S4": {"name": "Gryfów Śląski", "lat": 51.0307, "lon": 15.4268},
    "S5": {"name": "Świeradów-Zdrój", "lat": 50.9079, "lon": 15.3431},
    "S6": {"name": "Mirsk", "lat": 50.9699, "lon": 15.3866},
}


def normalize_name(value: str) -> str:
    simplified = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "", simplified)


def parse_dms(value: str) -> float:
    cleaned = value.replace("°", "*").replace("′", '"').replace("″", '"').replace("'", '"').replace("*", " ")
    match = re.search(
        r"(?P<deg>\d+)[^\d]+(?P<min>\d+)[^\d]+(?P<sec>[0-9.]+)[^\d]*(?P<hem>[NSEW])",
        cleaned,
    )
    if not match:
        raise ValueError(f"Nie mogę sparsować współrzędnej: {value}")
    deg = float(match.group("deg"))
    minutes = float(match.group("min"))
    seconds = float(match.group("sec"))
    hemisphere = match.group("hem").upper()
    decimal = deg + minutes / 60 + seconds / 3600
    if hemisphere in ("S", "W"):
        decimal *= -1
    return decimal


def extract_header_data(lines: Iterable[str]) -> Dict[str, str]:
    header = {}
    for line in lines:
        for key, pattern in HEADER_PATTERNS.items():
            match = pattern.search(line)
            if match:
                header[key] = match.group("value").strip()
    return header


def extract_header_from_list(lines: List[str]) -> Tuple[List[str], List[str]]:
    header_end_index = -1
    for i, line in enumerate(lines):
        l = line.lower()
        # Search for table header row
        if ("|lp.|" in l or " lp. " in l) and ("nazwa" in l or "hodowca" in l or "nazwisko" in l):
            header_end_index = i
            break
    
    if header_end_index == -1:
        # Fallback: look for the first line that looks like an arrival record
        for i, line in enumerate(lines):
            if re.search(r"\d-\d{2}:\d{2}:\d{2}", line) and re.match(r"^\s*\d+\s+", line):
                header_end_index = i
                break
    
    if header_end_index == -1:
        return lines, []
         
    return lines[:header_end_index], lines[header_end_index:]


def parse_section_stats_from_header(lines: Iterable[str]) -> List[Dict]:
    section_rows = []
    for line in lines:
        m = re.match(
            r"^\|\s*S(?P<section>\d+)\s+[^|]+\|\s*(?P<fanciers>\d+)\s*\|\s*(?P<sent>\d+)\s*\|\s*(?P<returned>\d+)\s*\|\s*(?P<pct_returned>[\d.]+)\s*\|\s*(?P<pct_birds>[\d.]+)\s*\|",
            line,
        )
        if not m:
            continue
        section = f"S{m.group('section')}"
        fanciers = int(m.group("fanciers"))
        sent = int(m.group("sent"))
        returned = int(m.group("returned"))
        pct_returned = float(m.group("pct_returned"))
        pct_birds = float(m.group("pct_birds"))
        loss_pct = ((sent - returned) / sent * 100) if sent else None
        section_rows.append(
            {
                "section": section,
                "fanciers": fanciers,
                "sent": sent,
                "returned": returned,
                "percent_returned": pct_returned,
                "percent_birds": pct_birds,
                "loss_pct": loss_pct,
            }
        )
    return section_rows


def derive_lofts_from_sections(arrivals: List[Dict], header: Dict) -> Dict:
    lofts = {}
    for entry in arrivals:
        key = entry["name_key"]
        if key not in lofts:
            lofts[key] = {
                "name": entry["name"],
                "section": entry["section"],
                "distance_km": entry["distance_km"],
                "birds_sent": entry.get("birds_sent_hint") or 0
            }
        else:
            if not lofts[key]["birds_sent"] and entry.get("birds_sent_hint"):
                lofts[key]["birds_sent"] = entry["birds_sent_hint"]
    
    for key, data in lofts.items():
        section_key = f"S{data['section']}"
        ref = DEFAULT_SECTION_REFERENCE.get(section_key)
        if ref:
            data["lat"] = ref["lat"]
            data["lon"] = ref["lon"]
        else:
            data["lat"] = 51.0
            data["lon"] = 15.5
            
    return lofts


def parse_arrivals(lines: Iterable[str], flight_date: dt.date) -> List[Dict]:
    arrivals = []
    birds_cache = {}
    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith('---') or 'KONIEC KONKURSU' in raw.upper() or raw.startswith('+--'):
            continue
        match = ARRIVAL_PATTERN.match(raw)
        if not match:
            continue
        data = match.groupdict()
        try:
            day_offset, time_part = data["arrival"].split("-")
            hours, minutes, seconds = map(int, time_part.split(":"))
            arrival_dt = dt.datetime.combine(
                flight_date, dt.time(hour=hours, minute=minutes, second=seconds)
            ) + dt.timedelta(days=int(day_offset) - 1)
        except Exception:
            continue

        birds_sent_hint = parse_birds_sent(data["wkm"])
        key = normalize_name(data["name"])
        if birds_sent_hint is not None:
            birds_cache[key] = birds_sent_hint
        else:
            birds_sent_hint = birds_cache.get(key)
            
        arrivals.append(
            {
                "lp": int(data["lp"]),
                "name": data["name"].strip(),
                "name_key": key,
                "section": data["section"],
                "ring": data["ring"],
                "category": data["category"],
                "typed": data["typed"] == "1",
                "birds_sent_hint": birds_sent_hint,
                "arrival_time": arrival_dt,
                "speed_mpm": float(data["speed"]),
                "distance_km": float(data["distance"]),
            }
        )
    return arrivals


def parse_birds_sent(token: str) -> Optional[int]:
    match = re.search(r"(\d+)", token)
    if not match:
        return None
    return int(match.group(1))


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def bearing(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> float:
    start_lat_rad, start_lon_rad = math.radians(start_lat), math.radians(start_lon)
    end_lat_rad, end_lon_rad = math.radians(end_lat), math.radians(end_lon)
    delta_lon = end_lon_rad - start_lon_rad
    x = math.sin(delta_lon) * math.cos(end_lat_rad)
    y = math.cos(start_lat_rad) * math.sin(end_lat_rad) - math.sin(start_lat_rad) * math.cos(end_lat_rad) * math.cos(delta_lon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def intermediate_point(lat1: float, lon1: float, lat2: float, lon2: float, f: float) -> Tuple[float, float]:
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    d = 2 * math.asin(math.sqrt(math.sin((lat2_rad - lat1_rad)/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin((lon2_rad - lon1_rad)/2)**2))
    if d == 0:
        return lat1, lon1
    A = math.sin((1 - f) * d) / math.sin(d)
    B = math.sin(f * d) / math.sin(d)
    x = A * math.cos(lat1_rad) * math.cos(lon1_rad) + B * math.cos(lat2_rad) * math.cos(lon2_rad)
    y = A * math.cos(lat1_rad) * math.sin(lon1_rad) + B * math.cos(lat2_rad) * math.sin(lon2_rad)
    z = A * math.sin(lat1_rad) + B * math.sin(lat2_rad)
    lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


def classify_wind(course_bearing: float, wind_direction: float) -> Tuple[str, float]:
    wind_to = (wind_direction + 180) % 360
    delta = ((wind_to - course_bearing + 540) % 360) - 180
    if abs(delta) <= 45: label = "wiatr w plecy"
    elif abs(delta) >= 135: label = "wiatr czołowy"
    elif delta > 0: label = "wiatr boczny z lewej"
    else: label = "wiatr boczny z prawej"
    return label, delta


HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "visibility",
    "precipitation",
    "rain",
    "snowfall",
    "showers",
    "shortwave_radiation",
    "uv_index",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "wind_speed_80m",
    "wind_direction_80m",
    "wind_speed_120m",
    "wind_direction_120m",
    "wind_speed_180m",
    "wind_direction_180m",
]


def fetch_weather_batch(lats, lons, start_time, end_time, session=None) -> List[Dict]:
    if not lats: return []
    global _OPENMETEO_DAILY_LIMIT_REACHED, _OPENMETEO_DAILY_LIMIT_LOGGED
    if _OPENMETEO_DAILY_LIMIT_REACHED:
        return []
    try:
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_hour": start_time.strftime("%Y-%m-%dT%H:%M"),
            "end_hour": end_time.strftime("%Y-%m-%dT%H:%M"),
            "windspeed_unit": "ms",
            "hourly": HOURLY_VARIABLES,
            "timezone": "Europe/Berlin",
        }
        try:
            responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        except Exception:
            params = {
                "latitude": lats,
                "longitude": lons,
                "start_date": start_time.date().isoformat(),
                "end_date": end_time.date().isoformat(),
                "windspeed_unit": "ms",
                "hourly": HOURLY_VARIABLES,
                "timezone": "Europe/Berlin",
            }
            responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        results = []
        for resp in responses:
            results.append({"hourly": parse_timeseries_response(resp)})
        return results
    except Exception as e:
        if "Daily API request limit exceeded" in str(e):
            _OPENMETEO_DAILY_LIMIT_REACHED = True
            if not _OPENMETEO_DAILY_LIMIT_LOGGED:
                _OPENMETEO_DAILY_LIMIT_LOGGED = True
                logging.warning("Open-Meteo: przekroczony dzienny limit; pomijam pogodę do końca uruchomienia.")
            return []
        logging.error("Błąd Open-Meteo: %s", e)
        return []


def parse_timeseries_response(response) -> List[Dict]:
    hourly = response.Hourly()
    utc_offset = response.UtcOffsetSeconds()
    start = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    freq = pd.Timedelta(seconds=hourly.Interval())
    timestamps = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    timestamps_local = (timestamps + pd.Timedelta(seconds=utc_offset)).tz_localize(None)
    data = {"time": timestamps_local.to_pydatetime()}
    for i, var_name in enumerate(HOURLY_VARIABLES):
        var = hourly.Variables(i)
        if var: data[var_name] = var.ValuesAsNumpy()
    df = pd.DataFrame(data).where(pd.notnull(pd.DataFrame(data)), None)
    return df.to_dict("records")


def build_report(header: Dict, arrivals: List[Dict], lofts: Dict, session: Optional[Session] = None, enable_weather: bool = True) -> List[Dict]:
    report = []
    fancier_arrivals = defaultdict(list)
    for a in arrivals: fancier_arrivals[a["name_key"]].append(a)
    
    release_dt = dt.datetime.strptime(f"{header['flight_date']} {header['release_time']}", "%d.%m.%Y %H:%M:%S")
    end_candidates = [a.get("arrival_time") for a in arrivals if a.get("arrival_time") is not None]
    flight_end_dt = max(end_candidates) if end_candidates else release_dt
    if flight_end_dt < release_dt:
        flight_end_dt = release_dt

    f_keys = list(fancier_arrivals.keys())
    keys_with_coords = [k for k in f_keys if k in lofts and lofts[k].get("lat") is not None and lofts[k].get("lon") is not None]
    coord_to_weather = {}
    release_lat = parse_dms(header.get("release_coords", "").split()[0])
    release_lon = parse_dms(header.get("release_coords", "").split()[1])
    release_coord = (round(float(release_lat), 5), round(float(release_lon), 5))
    # Precompute dystanse dla wspolczynnika osamotnienia
    keys_with_lofts = [k for k in f_keys if k in lofts and lofts[k].get("lat") is not None and lofts[k].get("lon") is not None]
    min_diff_map = {}
    avg_diff_map = {}
    for k in keys_with_lofts:
        dists_k = [haversine_distance(lofts[k]['lat'], lofts[k]['lon'], lofts[ok]['lat'], lofts[ok]['lon']) for ok in keys_with_lofts if ok != k]
        min_diff_map[k] = min(dists_k) if dists_k else 0.0
        avg_diff_map[k] = (sum(dists_k)/len(dists_k)) if dists_k else 0.0
    global_min_min = min(min_diff_map.values()) if min_diff_map else 0.0
    global_max_min = max(min_diff_map.values()) if min_diff_map else 0.0
    min_range = (global_max_min - global_min_min) if (global_max_min - global_min_min) > 0 else 1.0
    if enable_weather and not keys_with_coords:
        logging.warning("Brak współrzędnych gołębników; pomijam pogodę dla lotu.")
    if enable_weather and keys_with_coords:
        coords_in_order = []
        seen = set()
        # Always include release point
        if release_coord not in seen:
            seen.add(release_coord)
            coords_in_order.append(release_coord)
        # Include loft and mid points
        for k in keys_with_coords:
            lat = lofts[k]["lat"]
            lon = lofts[k]["lon"]
            loft_coord = (round(float(lat), 5), round(float(lon), 5))
            mid_lat, mid_lon = intermediate_point(release_lat, release_lon, lat, lon, 0.5)
            mid_coord = (round(float(mid_lat), 5), round(float(mid_lon), 5))
            for coord in (loft_coord, mid_coord):
                if coord not in seen:
                    seen.add(coord)
                    coords_in_order.append(coord)
        lats = [c[0] for c in coords_in_order]
        lons = [c[1] for c in coords_in_order]
        weather_list = fetch_weather_batch(lats, lons, release_dt, flight_end_dt + dt.timedelta(hours=1), session)
        if not weather_list:
            logging.warning("Nie udało się pobrać danych pogodowych dla lotu; kolumny pogodowe będą puste.")
        coord_to_weather = {coords_in_order[i]: weather_list[i] for i in range(min(len(coords_in_order), len(weather_list)))}
    
    for key, f_arrivals in fancier_arrivals.items():
        loft = lofts.get(key)
        if not loft: continue
        
        min_diff = min_diff_map.get(key, 0.0)
        avg_diff = avg_diff_map.get(key, 0.0)
        
        sent = loft.get("birds_sent", 0)
        classified = len(f_arrivals)
        loss = ((sent - classified)/sent * 100) if sent else 0
        avg_v = sum(a["speed_mpm"] for a in f_arrivals)/classified if classified else 0
        best_v = max(a["speed_mpm"] for a in f_arrivals) if f_arrivals else 0
        
        loft_coord = (round(float(loft["lat"]), 5), round(float(loft["lon"]), 5)) if loft.get("lat") is not None and loft.get("lon") is not None else None
        mid_lat, mid_lon = intermediate_point(release_lat, release_lon, loft['lat'], loft['lon'], 0.5) if loft.get("lat") is not None and loft.get("lon") is not None else (None, None)
        mid_coord = (round(float(mid_lat), 5), round(float(mid_lon), 5)) if mid_lat is not None and mid_lon is not None else None
        weather_release = coord_to_weather.get(release_coord)
        weather_mid = coord_to_weather.get(mid_coord) if mid_coord else None
        weather_loft = coord_to_weather.get(loft_coord) if loft_coord else None
        w_data = weather_loft
        w_row = None
        if w_data and w_data.get("hourly"):
            first_a = min((a.get("arrival_time") for a in f_arrivals if a.get("arrival_time") is not None), default=None)
            hourly_rows = [r for r in w_data["hourly"] if r.get("time") is not None]
            if hourly_rows:
                window_start = release_dt + dt.timedelta(minutes=30)
                window_end = first_a or flight_end_dt
                if window_end < window_start:
                    window_end = window_start
                window_rows = [r for r in hourly_rows if window_start <= r["time"] <= window_end]
                pool = window_rows or hourly_rows
                agg = {}
                for var in HOURLY_VARIABLES:
                    vals = [r.get(var) for r in pool if r.get(var) is not None]
                    agg[var] = float(pd.Series(vals).median()) if vals else None
                w_row = agg
        
        course = bearing(release_lat, release_lon, loft['lat'], loft['lon'])
        wind_label, wind_delta = (None, None)
        wind_proj = None
        wind_proj_80m = None
        wind_proj_120m = None
        wind_proj_180m = None
        # Trójpunktowy model wiatru (A: start, B: środek, C: meta) z wagami Simpsona
        wind_effective_m_s = None
        wind_label_from_tripoint = None
        def nearest_row(weather, target_time):
            if not weather or not weather.get("hourly") or target_time is None:
                return None
            rows = [r for r in weather["hourly"] if r.get("time") is not None]
            if not rows: return None
            return min(rows, key=lambda r: abs((r["time"] - target_time).total_seconds()))
        first_a = min((a.get("arrival_time") for a in f_arrivals if a.get("arrival_time") is not None), default=None)
        tA = release_dt
        tC = first_a or flight_end_dt
        tB = tA + (tC - tA)/2 if tC and tA else None
        rowA = nearest_row(weather_release, tA)
        rowB = nearest_row(weather_mid, tB)
        rowC = nearest_row(weather_loft, tC)
        samples = []
        for w_row_i, weight in ((rowA, 1.0), (rowB, 4.0), (rowC, 1.0)):
            if w_row_i and w_row_i.get("wind_direction_10m") is not None and w_row_i.get("wind_speed_10m") is not None:
                to_dir = (w_row_i["wind_direction_10m"] + 180) % 360
                rad = math.radians(to_dir)
                u = w_row_i["wind_speed_10m"] * math.sin(rad)
                v = w_row_i["wind_speed_10m"] * math.cos(rad)
                samples.append((u, v, weight))
        if samples:
            wsum = sum(w for _, _, w in samples)
            u_eff = sum(u*w for u, v, w in samples) / wsum
            v_eff = sum(v*w for u, v, w in samples) / wsum
            cr = math.radians(course)
            u_course = math.sin(cr)
            v_course = math.cos(cr)
            wind_effective_m_s = u_eff * u_course + v_eff * v_course
            dir_eff_rad = math.atan2(u_eff, v_eff)
            dir_eff_deg = (math.degrees(dir_eff_rad) + 360) % 360
            wind_label_from_tripoint, _ = classify_wind(course, (dir_eff_deg - 180) % 360)

        turbulencja = None
        if w_row and w_row.get("wind_gusts_10m") is not None and w_row.get("wind_speed_10m") is not None:
            turbulencja = w_row["wind_gusts_10m"] - w_row["wind_speed_10m"]
        
        # Czas rozrzutu u hodowcy (1. do 5. lub ostatniego)
        times = sorted([a.get("arrival_time") for a in f_arrivals if a.get("arrival_time") is not None])
        rozrzut_min = 0.0
        if times:
            idx = min(4, len(times) - 1)
            rozrzut_min = (times[idx] - times[0]).total_seconds() / 60.0
        
        # Współczynnik osamotnienia (0..1) na bazie globalnych min dystansów
        wsp_osamotnienia = (min_diff - global_min_min) / min_range if min_range > 0 else 0.0
        
        # Ocena trudności pogody i straty skorygowane
        w_eff_kmh = (wind_effective_m_s * 3.6) if wind_effective_m_s is not None else None
        precipitation_mm = w_row.get("precipitation") if w_row else None
        turb_kmh = (turbulencja * 3.6) if turbulencja is not None else None
        ocena_trudnosci = 1.0
        if w_eff_kmh is not None:
            if w_eff_kmh < 0:
                ocena_trudnosci += abs(w_eff_kmh) / 10.0
            else:
                ocena_trudnosci -= min(0.4, w_eff_kmh / 20.0)
        if precipitation_mm is not None:
            ocena_trudnosci += 0.3 * float(precipitation_mm)
        if turb_kmh is not None:
            ocena_trudnosci += 0.05 * float(turb_kmh)
        if ocena_trudnosci < 0.5:
            ocena_trudnosci = 0.5
        if ocena_trudnosci > 5.0:
            ocena_trudnosci = 5.0
        straty_skorygowane = (loss / ocena_trudnosci) if ocena_trudnosci else loss

        report.append({
            "hodowca": loft["name"], "sekcja": loft.get("section"),
            "ptaki_wyslane": sent, "ptaki_klasyfikowane": classified,
            "straty_min_proc": f"{loss:.2f}", "srednia_predkosc": f"{avg_v:.2f}",
            "najlepsza_predkosc": f"{best_v:.2f}", "kurs_stopnie": f"{course:.2f}",
            "wiatr_opis": wind_label_from_tripoint or wind_label,
            "wiatr_proj_10m": f"{wind_proj:.2f}" if wind_proj is not None else None,
            "wiatr_efektywny_kmh": f"{(wind_effective_m_s*3.6):.2f}" if wind_effective_m_s is not None else None,
            "wiatr_proj_80m": f"{wind_proj_80m:.2f}" if wind_proj_80m is not None else None,
            "wiatr_proj_120m": f"{wind_proj_120m:.2f}" if wind_proj_120m is not None else None,
            "wiatr_proj_180m": f"{wind_proj_180m:.2f}" if wind_proj_180m is not None else None,
            "temperatura_C": w_row.get("temperature_2m") if w_row else None,
            "cisnienie_hPa": w_row.get("pressure_msl") if w_row else None,
            "opady_mm": w_row.get("precipitation") if w_row else None,
            "wilgotnosc": w_row.get("relative_humidity_2m") if w_row else None,
            "zachmurzenie": w_row.get("cloud_cover") if w_row else None,
            "widocznosc": w_row.get("visibility") if w_row else None,
            "deszcz": w_row.get("rain") if w_row else None,
            "snieg": w_row.get("snowfall") if w_row else None,
            "przelotny_deszcz": w_row.get("showers") if w_row else None,
            "cisnienie_powierzchniowe": w_row.get("surface_pressure") if w_row else None,
            "promieniowanie_sloneczne": w_row.get("shortwave_radiation") if w_row else None,
            "indeks_uv": w_row.get("uv_index") if w_row else None,
            "wskaznik_turbulencji": f"{turbulencja:.2f}" if turbulencja is not None else None,
            "roznica_odleglosci_min_km": f"{min_diff:.2f}",
            "roznica_odleglosci_srednia_km": f"{avg_diff:.2f}",
            "rozrzut_czasu_min": f"{rozrzut_min:.2f}",
            "wspolczynnik_osamotnienia": f"{wsp_osamotnienia:.2f}",
            "ocena_trudnosci_pogody": f"{ocena_trudnosci:.2f}",
            "straty_skorygowane_pogoda": f"{straty_skorygowane:.2f}",
        })
    return report


def persist_flight_data(flight_id, header, section_stats, arrivals, report_rows, lofts, release_lat, release_lon, lot_dir, raw_lines, header_lines, flight_date_iso, report_output) -> None:
    ensure_storage_dirs()
    lot_dir.mkdir(parents=True, exist_ok=True)
    (lot_dir / "list.txt").write_text("\n".join(raw_lines), encoding="utf-8")
    (lot_dir / "header.txt").write_text("\n".join(header_lines), encoding="utf-8")
    
    df_report = pd.DataFrame(report_rows)
    if "fancier_id" in df_report.columns:
        df_report = df_report.drop(columns=["fancier_id"])
    expected_cols = [
        "hodowca", "sekcja",
        "ptaki_wyslane", "ptaki_klasyfikowane",
        "straty_min_proc", "srednia_predkosc", "najlepsza_predkosc",
        "kurs_stopnie",
        "wiatr_opis", "wiatr_proj_10m", "wiatr_efektywny_kmh", "wiatr_proj_80m", "wiatr_proj_120m", "wiatr_proj_180m",
        "temperatura_C", "cisnienie_hPa", "opady_mm", "wilgotnosc", "zachmurzenie", "widocznosc",
        "deszcz", "snieg", "przelotny_deszcz", "cisnienie_powierzchniowe", "promieniowanie_sloneczne", "indeks_uv",
        "wskaznik_turbulencji",
        "roznica_odleglosci_min_km", "roznica_odleglosci_srednia_km",
        "rozrzut_czasu_min", "wspolczynnik_osamotnienia", "ocena_trudnosci_pogody", "straty_skorygowane_pogoda",
    ]
    for c in expected_cols:
        if c not in df_report.columns:
            df_report[c] = None
    df_report = df_report[expected_cols]
    df_report.to_csv(report_output, index=False, encoding="utf-8")
    
    f_row = [{
        "flight_id": flight_id, "season": header.get("season", ""),
        "list_name": header.get("list_name", ""), "flight_date": flight_date_iso,
        "release_time": header.get("release_time", ""), "release_point": header.get("release_point", ""),
        "release_lat": f"{release_lat:.6f}", "release_lon": f"{release_lon:.6f}",
        "fanciers": header.get("fanciers", ""), "birds_sent": header.get("birds_sent", ""),
        "birds_returned": header.get("birds_returned", ""), "avg_distance_km": header.get("avg_distance", ""),
        "lot_dir": str(lot_dir)
    }]
    append_rows(FLIGHTS_CSV, FLIGHTS_FIELDNAMES, f_row, lambda r: r.get("flight_id") == flight_id)

    arrivals_rows = []
    for a in arrivals:
        arrivals_rows.append(
            {
                "flight_id": flight_id,
                "lp": a.get("lp"),
                "fancier_id": a.get("name_key"),
                "name": a.get("name"),
                "ring": a.get("ring"),
                "category": a.get("category"),
                "typed": 1 if a.get("typed") else 0,
                "arrival_time": a.get("arrival_time").isoformat() if a.get("arrival_time") else None,
                "speed_mpm": a.get("speed_mpm"),
                "distance_km": a.get("distance_km"),
            }
        )
    append_rows(ARRIVALS_CSV, ARRIVALS_FIELDNAMES, arrivals_rows, lambda r: r.get("flight_id") == flight_id)

    existing_fanciers = {}
    if FANCIERS_CSV.exists():
        with FANCIERS_CSV.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("fancier_id"):
                    existing_fanciers[r["fancier_id"]] = r
    merged_fanciers = dict(existing_fanciers)
    for f_id, loft in lofts.items():
        section = loft.get("section")
        section_norm = f"S{section}" if section and not str(section).startswith("S") else section
        row = {
            "fancier_id": f_id,
            "name": loft.get("name"),
            "section": section_norm,
            "lat": loft.get("lat"),
            "lon": loft.get("lon"),
            "default_birds_sent": loft.get("birds_sent") or 0,
        }
        prev = merged_fanciers.get(f_id)
        if not prev:
            merged_fanciers[f_id] = row
        else:
            lat_prev = prev.get("lat")
            lon_prev = prev.get("lon")
            if (not lat_prev or not lon_prev) and row.get("lat") is not None and row.get("lon") is not None:
                prev["lat"] = row["lat"]
                prev["lon"] = row["lon"]
            if not prev.get("section") and row.get("section"):
                prev["section"] = row["section"]
            if (not prev.get("default_birds_sent") or str(prev.get("default_birds_sent")) == "0") and row.get("default_birds_sent"):
                prev["default_birds_sent"] = row["default_birds_sent"]
            if not prev.get("name") and row.get("name"):
                prev["name"] = row["name"]
            merged_fanciers[f_id] = prev
    with FANCIERS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FANCIERS_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged_fanciers.values())
    results_rows = []
    for rr in report_rows:
        name = rr.get("hodowca")
        fancier_id = normalize_name(name) if name else rr.get("fancier_id")
        section = rr.get("sekcja")
        section_norm = f"S{section}" if section and not str(section).startswith("S") else section
        birds_sent = rr.get("ptaki_wyslane")
        birds_classified = rr.get("ptaki_klasyfikowane")
        loss_min_pct = rr.get("straty_min_proc")
        avg_speed = rr.get("srednia_predkosc")
        best_speed = rr.get("najlepsza_predkosc")
        course_deg = rr.get("kurs_stopnie")
        wind_label = rr.get("wiatr_opis")
        wind_proj_10m = rr.get("wiatr_proj_10m")
        temperature_C = rr.get("temperatura_C")
        pressure_hPa = rr.get("cisnienie_hPa")
        precipitation_mm = rr.get("opady_mm")
        results_rows.append(
            {
                "flight_id": flight_id,
                "fancier_id": fancier_id,
                "name": name,
                "section": section_norm,
                "birds_sent": birds_sent,
                "birds_classified": birds_classified,
                "loss_min_pct": loss_min_pct,
                "avg_speed": avg_speed,
                "best_speed": best_speed,
                "course_deg": course_deg,
                "wind_label": wind_label,
                "wind_proj_m_s": wind_proj_10m,
                "wind_proj_10m": wind_proj_10m,
                "turbulence_index_10m": rr.get("wskaznik_turbulencji"),
                "wind_effective_10m_kmh": rr.get("wiatr_efektywny_kmh"),
                "wind_proj_80m": rr.get("wiatr_proj_80m"),
                "wind_proj_100m": None,
                "wind_proj_120m": rr.get("wiatr_proj_120m"),
                "wind_proj_180m": rr.get("wiatr_proj_180m"),
                "temperature_C": temperature_C,
                "pressure_hPa": pressure_hPa,
                "precipitation_mm": precipitation_mm,
                "humidity_pct": rr.get("wilgotnosc"),
                "cloud_cover_pct": rr.get("zachmurzenie"),
                "visibility_m": rr.get("widocznosc"),
                "rain_mm": rr.get("deszcz"),
                "snowfall_mm": rr.get("snieg"),
                "showers_mm": rr.get("przelotny_deszcz"),
                "surface_pressure_hPa": rr.get("cisnienie_powierzchniowe"),
                "shortwave_radiation": rr.get("promieniowanie_sloneczne"),
                "uv_index": rr.get("indeks_uv"),
            }
        )
    append_rows(RESULTS_CSV, RESULTS_FIELDNAMES, results_rows, lambda r: r.get("flight_id") == flight_id)

    sections_path = DATA_ROOT / "sections.csv"
    if section_stats:
        sec_rows = []
        for s in section_stats:
            sec_rows.append({"flight_id": flight_id, **s})
        append_rows(sections_path, SECTIONS_FIELDNAMES, sec_rows, lambda r: r.get("flight_id") == flight_id)


def ensure_storage_dirs():
    for d in [LOTS_ROOT, DB_ROOT]: d.mkdir(parents=True, exist_ok=True)


def append_rows(path, fieldnames, new_rows, remove_condition=None):
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if not (remove_condition and remove_condition(r))]
    rows.extend(new_rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_csv(path: Path, fieldnames: List[str]) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    for r in rows:
        for fn in fieldnames:
            if fn not in r:
                r[fn] = None
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_db() -> None:
    ensure_storage_dirs()
    normalize_csv(FLIGHTS_CSV, FLIGHTS_FIELDNAMES)
    normalize_csv(ARRIVALS_CSV, ARRIVALS_FIELDNAMES)
    normalize_csv(FANCIERS_CSV, FANCIERS_FIELDNAMES)
    normalize_csv(RESULTS_CSV, RESULTS_FIELDNAMES)
    normalize_csv(SECTIONS_CSV, SECTIONS_FIELDNAMES)


def slugify(value: str) -> str:
    v = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", v).strip("-")


def build_lot_id(header: Dict) -> str:
    d = dt.datetime.strptime(header["flight_date"], "%d.%m.%Y").date()
    slug = slugify(header.get("list_name") or header.get("release_point") or "lot")
    return f"{d.isoformat()}-{slug}"


def process_flight(list_path=None, download_url=None, force_rebuild=False, session=None, enable_weather: bool = True, logger=None, **kwargs) -> Dict:
    if download_url:
        if not session: session = requests.Session()
        resp = session.get(download_url)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as tmp:
            tmp.write(resp.text)
            list_path = Path(tmp.name)
    
    raw = list_path.read_text(encoding="utf-8").splitlines()
    h_lines, b_lines = extract_header_from_list(raw)
    header = extract_header_data(h_lines)
    if not header.get("flight_date"): raise ValueError("Brak daty lotu w nagłówku.")
    section_stats = parse_section_stats_from_header(h_lines)
    
    f_id = build_lot_id(header)
    lot_dir = LOTS_ROOT / f_id
    out = lot_dir / "report.csv"
    
    if out.exists() and not force_rebuild:
        return {"flight_id": f_id, "report_path": str(out), "flight_date": header["flight_date"], "release_point": header.get("release_point")}
        
    arrivals = parse_arrivals(b_lines, dt.datetime.strptime(header["flight_date"], "%d.%m.%Y").date())
    lofts = derive_lofts_from_sections(arrivals, header)
    report = build_report(header, arrivals, lofts, session=session, enable_weather=enable_weather)
    
    r_lat = parse_dms(header.get("release_coords", "").split()[0])
    r_lon = parse_dms(header.get("release_coords", "").split()[1])
    
    persist_flight_data(f_id, header, section_stats, arrivals, report, lofts, r_lat, r_lon, lot_dir, raw, h_lines, dt.datetime.strptime(header["flight_date"], "%d.%m.%Y").date().isoformat(), out)
    
    if download_url: os.remove(list_path)
    return {"flight_id": f_id, "report_path": str(out), "flight_date": header["flight_date"], "release_point": header.get("release_point")}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list-path", type=Path)
    parser.add_argument("-u", "--url")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--normalize-db", action="store_true")
    args = parser.parse_args()
    try:
        if args.normalize_db:
            normalize_db()
            print("Sukces! Baza danych została ujednolicona.")
            return
        res = process_flight(list_path=args.list_path, download_url=args.url, enable_weather=not args.no_weather, force_rebuild=args.force)
        print(f"Sukces! Raport: {res['report_path']}")
    except Exception as e:
        print(f"Błąd: {e}")

if __name__ == "__main__":
    main()
