# Open‑Meteo – API Documentation (sklejone)

Ten plik zawiera sklejone fragmenty dokumentacji, które wkleiłeś w rozmowie, w jednym miejscu.

## Endpoint: /v1/forecast

`/v1/forecast` accepts a geographical coordinate, a list of weather variables and responds with a JSON hourly weather forecast for 7 days. Time always starts at 0:00 today and contains 168 hours. If `&forecast_days=16` is set, up to 16 days of forecast can be returned.

### Parameters

| Parameter | Format | Required | Default | Description |
|---|---:|:---:|---:|---|
| latitude, longitude | Floating point | Yes |  | Geographical WGS84 coordinates of the location. Multiple coordinates can be comma separated. E.g. `&latitude=52.52,48.85&longitude=13.41,2.35`. For North and South America locations use negative longitudes. |
| elevation | Floating point | No |  | Elevation used for statistical downscaling. Default: 90 meter DEM. `&elevation=nan` disables downscaling. Multiple locations: comma separated. |
| hourly | String array | No |  | Weather variables returned. Values can be comma separated, or multiple `&hourly=` params. |
| daily | String array | No |  | Daily aggregations. If daily specified, `timezone` is required. |
| current | String array | No |  | Current conditions variables. |
| temperature_unit | String | No | celsius | If `fahrenheit` is set, values are converted to °F. |
| wind_speed_unit | String | No | kmh | Other units: `ms`, `mph`, `kn`. |
| precipitation_unit | String | No | mm | Other unit: `inch`. |
| timeformat | String | No | iso8601 | If `unixtime`, timestamps are UNIX seconds (GMT+0); for daily with unix timestamps apply `utc_offset_seconds` again. |
| timezone | String | No | GMT | If set, all timestamps are returned as local-time and data starts at 00:00 local-time. `auto` resolves timezone from coordinates. Multiple coordinates: comma-separated list. |
| past_days | Integer (0-92) | No | 0 | Includes yesterday / days before yesterday. |
| forecast_days | Integer (0-16) | No | 7 | Default 7 days, max 16. |
| forecast_hours / forecast_minutely_15 / past_hours / past_minutely_15 | Integer (>0) | No |  | Controls number of timesteps; uses current hour / 15-min step as reference. |
| start_date / end_date | String (yyyy-mm-dd) | No |  | Interval to get weather data. |
| start_hour / end_hour / start_minutely_15 / end_minutely_15 | String (yyyy-mm-ddThh:mm) | No |  | Interval for hourly or 15-minutely data. |
| models | String array | No | auto | Select weather models. |
| cell_selection | String | No | land | Grid-cell selection preference: `land`, `sea`, `nearest`. |
| apikey | String | No |  | Only required for commercial use to access reserved resources. |

## Endpoint: /v1/archive

The API endpoint `/v1/archive` allows users to retrieve historical weather data for a specific location and time period. To use this endpoint, you can specify a geographical coordinate, a time interval, and a list of weather variables that they are interested in. The endpoint will then return the requested data in a format that can be easily accessed and used by applications or other software. This endpoint can be very useful for researchers and other users who need to access detailed historical weather data for specific locations and time periods.

All URL parameters are listed below. Additional optional URL parameters will be added. For API stability, no required parameters will be added in the future.

### Parameters

| Parameter | Format | Required | Default | Description |
|---|---:|:---:|---:|---|
| latitude, longitude | Floating point | Yes |  | Geographical WGS84 coordinates of the location. Multiple coordinates can be comma separated. E.g. `&latitude=52.52,48.85&longitude=13.41,2.35`. To return data for multiple locations the JSON output changes to a list of structures. CSV and XLSX formats add a column `location_id`. |
| elevation | Floating point | No |  | The elevation used for statistical downscaling. Per default, a 90 meter digital elevation model is used. You can manually set the elevation to correctly match mountain peaks. If `&elevation=nan` is specified, downscaling will be disabled and the API uses the average grid-cell height. For multiple locations, elevation can also be comma separated. |
| start_date / end_date | String (yyyy-mm-dd) | Yes |  | The time interval to get weather data. A day must be specified as an ISO8601 date (e.g. 2022-12-31). |
| hourly | String array | No |  | A list of weather variables which should be returned. Values can be comma separated, or multiple `&hourly=` parameter in the URL can be used. |
| daily | String array | No |  | A list of daily weather variable aggregations which should be returned. Values can be comma separated, or multiple `&daily=` parameter in the URL can be used. If daily weather variables are specified, parameter `timezone` is required. |
| temperature_unit | String | No | celsius | If `fahrenheit` is set, all temperature values are converted to Fahrenheit. |
| wind_speed_unit | String | No | kmh | Other wind speed speed units: `ms`, `mph` and `kn`. |
| precipitation_unit | String | No | mm | Other precipitation amount units: `inch`. |
| timeformat | String | No | iso8601 | If format `unixtime` is selected, all time values are returned in UNIX epoch time in seconds. Please note that all time is then in GMT+0! For daily values with unix timestamp, please apply `utc_offset_seconds` again to get the correct date. |
| timezone | String | No | GMT | If timezone is set, all timestamps are returned as local-time and data is returned starting at 00:00 local-time. Any time zone name from the time zone database is supported. If `auto` is set as a time zone, the coordinates will be automatically resolved to the local time zone. For multiple coordinates, a comma separated list of timezones can be specified. |
| cell_selection | String | No | land | Set a preference how grid-cells are selected. The default `land` finds a suitable grid-cell on land with similar elevation to the requested coordinates using a 90-meter digital elevation model. `sea` prefers grid-cells on sea. `nearest` selects the nearest possible grid-cell. |
| apikey | String | No |  | Only required to commercial use to access reserved API resources for customers. The server URL requires the prefix `customer-`. See pricing for more information. |

### Hourly Parameter Definition

The parameter `&hourly=` accepts the following values. Most weather variables are given as an instantaneous value for the indicated hour. Some variables like precipitation are calculated from the preceding hour as and average or sum.

Selected examples you provided:

| Variable | Valid time | Unit | Description |
|---|---|---|---|
| temperature_2m | Instant | °C (°F) | Air temperature at 2 meters above ground |
| relative_humidity_2m | Instant | % | Relative humidity at 2 meters above ground |
| dew_point_2m | Instant | °C (°F) | Dew point temperature at 2 meters above ground |
| apparent_temperature | Instant | °C (°F) | Apparent temperature is the perceived feels-like temperature combining wind chill factor, relative humidity and solar radiation |
| pressure_msl / surface_pressure | Instant | hPa | Atmospheric air pressure reduced to mean sea level (msl) or pressure at surface. Typically pressure on mean sea level is used in meteorology. Surface pressure gets lower with increasing elevation. |
| precipitation | Preceding hour sum | mm (inch) | Total precipitation (rain, showers, snow) sum of the preceding hour. Data is stored with a 0.1 mm precision. |
| rain | Preceding hour sum | mm (inch) | Only liquid precipitation of the preceding hour including local showers and rain from large scale systems. |
| snowfall | Preceding hour sum | cm (inch) | Snowfall amount of the preceding hour in centimeters. For the water equivalent in millimeter, divide by 7. E.g. 7 cm snow = 10 mm precipitation water equivalent |
| cloud_cover | Instant | % | Total cloud cover as an area fraction |
| cloud_cover_low | Instant | % | Low level clouds and fog up to 2 km altitude |
| cloud_cover_mid | Instant | % | Mid level clouds from 2 to 6 km altitude |
| cloud_cover_high | Instant | % | High level clouds from 6 km altitude |
| shortwave_radiation | Preceding hour mean | W/m² | Shortwave solar radiation as average of the preceding hour. This is equal to the total global horizontal irradiation |
| direct_radiation / direct_normal_irradiance | Preceding hour mean | W/m² | Direct solar radiation as average of the preceding hour on the horizontal plane and the normal plane (perpendicular to the sun) |
| diffuse_radiation | Preceding hour mean | W/m² | Diffuse solar radiation as average of the preceding hour |
| global_tilted_irradiance | Preceding hour mean | W/m² | Total radiation received on a tilted pane as average of the preceding hour. |

## Run Your Own Weather API

If you require an extensive amount of weather data through an API daily and wish to run your own weather API, you can obtain current weather data from Open-Meteo on AWS Open Data. The Open-Meteo Docker container can listen for newly published data and keep your local database up-to-date.

Basic steps include:

- Install the Open-Meteo Docker image
- Start the data synchronization for given weather models
- Launch the API instance and get the latest forecast from your new API endpoint

Repository: https://github.com/open-meteo/open-data

### 1. Install Open‑Meteo Docker image

```bash
docker pull ghcr.io/open-meteo/open-meteo
```

Create a Docker volume to store weather data:

```bash
docker volume create --name open-meteo-data
```

### 2. Download Weather Forecasts

Download the digital elevation model (DEM):

```bash
docker run -it --rm -v open-meteo-data:/app/data ghcr.io/open-meteo/open-meteo sync copernicus_dem90 static
```

Download GFS temperature (2m) and include past days:

```bash
docker run -it --rm -v open-meteo-data:/app/data ghcr.io/open-meteo/open-meteo sync ncep_gfs013 temperature_2m --past-days 3
```

Check every 10 minutes for updated forecasts (background):

```bash
docker run -d --rm -v open-meteo-data:/app/data ghcr.io/open-meteo/open-meteo sync ncep_gfs013 temperature_2m --past-days 3 --repeat-interval 10
```

### 3. Start API endpoint

Start the API service on http://127.0.0.1:8080:

```bash
docker run -d --rm -v open-meteo-data:/app/data -p 8080:8080 ghcr.io/open-meteo/open-meteo
```

Fetch a forecast:

```bash
curl "http://127.0.0.1:8080/v1/forecast?latitude=47.1&longitude=8.4&models=gfs_global&hourly=temperature_2m"
```

### docker-compose

The repository contains a `docker-compose.yml` to automate the steps above. Example workflow:

```bash
git clone https://github.com/open-meteo/open-data open-meteo
cd open-meteo
docker compose up -d
docker compose down
```

## Hourly Parameter Definition

`&hourly=` accepts the following values. Most weather variables are given as an instantaneous value for the indicated hour. Some variables like precipitation are calculated from the preceding hour as an average or sum.

Examples you provided:

| Variable | Valid time | Unit | Description |
|---|---|---|---|
| temperature_2m | Instant | °C (°F) | Air temperature at 2 meters above ground |
| relative_humidity_2m | Instant | % | Relative humidity at 2 meters above ground |
| pressure_msl / surface_pressure | Instant | hPa | Atmospheric air pressure reduced to mean sea level (msl) or pressure at surface |
| cloud_cover | Instant | % | Total cloud cover |
| wind_speed_10m / 80m / 120m / 180m | Instant | km/h (mph, m/s, knots) | Wind speed at 10/80/120/180m |
| wind_direction_10m / 80m / 120m / 180m | Instant | ° | Wind direction at 10/80/120/180m |
| wind_gusts_10m | Preceding hour max | km/h (mph, m/s, knots) | Gusts at 10m |
| shortwave_radiation | Preceding hour mean | W/m² | Shortwave solar radiation |
| precipitation | Preceding hour sum | mm (inch) | Total precipitation (rain, showers, snow) sum of preceding hour |
| rain | Preceding hour sum | mm (inch) | Rain from large scale weather systems |
| showers | Preceding hour sum | mm (inch) | Showers from convective precipitation |
| snowfall | Preceding hour sum | cm (inch) | Snowfall amount in centimeters (water equiv: divide by 7) |
| visibility | Instant | meters | Viewing distance in meters |
| uv_index | (varies) | Index | UV index |

## 15‑Minutely Parameter Definition (minutely_15)

The parameter `&minutely_15=` can be used to get 15-minutely data. This data is based on NOAA HRRR model for North America and DWD ICON-D2 and Météo-France AROME model for Central Europe. If 15-minutely data is requested for other regions data is interpolated from 1-hourly to 15-minutely.

Selected examples you provided:

| Variable | Valid time | Unit | Notes |
|---|---|---|---|
| shortwave_radiation | Preceding 15 minutes mean | W/m² |  |
| direct_radiation / direct_normal_irradiance | Preceding 15 minutes mean | W/m² | Direct solar radiation |
| diffuse_radiation | Preceding 15 minutes mean | W/m² | Diffuse solar radiation |
| precipitation / rain / showers | Preceding 15 minutes sum | mm (inch) |  |
| wind_speed_10m / wind_speed_80m | Instant | km/h (mph, m/s, knots) |  |
| wind_direction_10m / wind_direction_80m | Instant | ° |  |
| visibility | Instant | meters |  |
| weather_code | Instant | WMO code |  |

## Pressure Level Variables

Pressure level variables do not have fixed altitudes. Altitude varies with atmospheric pressure. 1000 hPa is roughly between 60 and 160 meters above sea level. For precise altitudes, `geopotential_height` can be used. Altitudes are in meters above sea level (not above ground).

Examples you provided:

| Variable | Unit | Description |
|---|---|---|
| temperature_1000hPa, temperature_975hPa, ... | °C (°F) | Air temperature at the specified pressure level |
| relative_humidity_1000hPa, relative_humidity_975hPa, ... | % | Relative humidity at the specified pressure level |
| dew_point_1000hPa, dew_point_975hPa, ... | °C (°F) | Dew point temperature at the specified pressure level |
| cloud_cover_1000hPa, cloud_cover_975hPa, ... | % | Cloud cover at the specified pressure level |
| wind_speed_1000hPa, wind_speed_975hPa, ... | km/h (mph, m/s, knots) | Wind speed at the specified pressure level |
| wind_direction_1000hPa, wind_direction_975hPa, ... | ° | Wind direction at the specified pressure level |
| geopotential_height_1000hPa, geopotential_height_975hPa, ... | meter | Geopotential height to get correct altitude above sea level |

## JSON Return Object

On success a JSON object will be returned.

```json
{
  "latitude": 52.52,
  "longitude": 13.419,
  "elevation": 44.812,
  "generationtime_ms": 2.2119,
  "utc_offset_seconds": 0,
  "timezone": "Europe/Berlin",
  "timezone_abbreviation": "CEST",
  "hourly": {
    "time": ["2022-07-01T00:00", "2022-07-01T01:00", "2022-07-01T02:00"],
    "temperature_2m": [13, 12.7, 12.7]
  },
  "hourly_units": {
    "temperature_2m": "°C"
  }
}
```

Additional fields description you provided:

- `latitude, longitude`: WGS84 of the center of the weather grid-cell used to generate the forecast (can differ from requested coordinate by a few km).
- `elevation`: elevation from 90m DEM (affects grid-cell selection and downscaling).
- `generationtime_ms`: generation time (performance monitoring).
- `utc_offset_seconds`: applied timezone offset from `&timezone=`.
- `timezone`, `timezone_abbreviation`: timezone identifier and abbreviation.
- `current`: object of chosen current variables; `time` indicates validity moment; `interval` is duration in seconds used for backward-looking sums/averages (e.g. 900 seconds for 15-min).
- `hourly`: arrays for each selected variable + `time`.
- `hourly_units`: unit per hourly variable.
- `daily`, `daily_units`: daily aggregates and their units.

## Errors

If an error occurs (e.g. invalid parameter), a JSON error object is returned with HTTP 400.

```json
{
  "error": true,
  "reason": "Cannot initialize WeatherVariable from invalid String value tempeture_2m for key hourly"
}
```

## WMO Weather interpretation codes (WW)

| Code | Description |
|---:|---|
| 0 | Clear sky |
| 1, 2, 3 | Mainly clear, partly cloudy, and overcast |
| 45, 48 | Fog and depositing rime fog |
| 51, 53, 55 | Drizzle: Light, moderate, and dense intensity |
| 56, 57 | Freezing Drizzle: Light and dense intensity |
| 61, 63, 65 | Rain: Slight, moderate and heavy intensity |
| 66, 67 | Freezing Rain: Light and heavy intensity |
| 71, 73, 75 | Snow fall: Slight, moderate, and heavy intensity |
| 77 | Snow grains |
| 80, 81, 82 | Rain showers: Slight, moderate, and violent |
| 85, 86 | Snow showers slight and heavy |
| 95* | Thunderstorm: Slight or moderate |
| 96, 99* | Thunderstorm with slight and heavy hail |

(*) Thunderstorm forecast with hail is only available in Central Europe.

## OpenWeatherMap – History API (porównanie)

OpenWeatherMap udostępnia godzinowe historyczne dane pogodowe dla dowolnej lokalizacji przez History API. Dostępność danych zależy od rodzaju subskrypcji. Dane można pobrać w formacie JSON lub CSV (w dokumentacji są też warianty „History Bulk” i „History Forecast Bulk”).

### Endpoint: Hourly History

Wywołanie API (wariant z `start` i `end`):

`https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={API_key}`

Wywołanie API (wariant z `start` i `cnt`):

`https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&cnt={cnt}&appid={API_key}`

Przykład:

`https://history.openweathermap.org/data/2.5/history/city?lat=41.85&lon=-87.65&appid={API_key}`

### Parametry

| Parametr | Wymagany | Opis |
|---|:---:|---|
| lat | tak | Szerokość geograficzna. Do konwersji miasto/kod ↔ współrzędne użyj OpenWeatherMap Geocoding API. |
| lon | tak | Długość geograficzna. |
| type | tak | Typ wywołania — należy zostawić `hour`. |
| appid | tak | Klucz API. Brak/niepoprawny klucz zwraca np. `401` z komunikatem „Invalid API key”. |
| start | nie | Czas startu (Unix time, UTC), np. `start=1369728000`. |
| end | nie | Czas końca (Unix time, UTC), np. `end=1369789200`. |
| cnt | nie | Liczba znaczników czasu w odpowiedzi (1 na godzinę), można użyć zamiast `end`. |

### Pola w odpowiedzi (skrót)

- `cod`, `message`, `calctime`: pola wewnętrzne
- `city_id`: identyfikator miasta
- `list[]`: lista rekordów godzinowych
  - `dt`: czas (Unix, UTC)
  - `main.temp`, `main.feels_like`, `main.pressure`, `main.humidity`, `main.sea_level`, `main.grnd_level`, `main.temp_min`, `main.temp_max`
  - `wind.speed`, `wind.deg`
  - `clouds.all`
  - `rain.1h`, `rain.3h`
  - `snow.1h`, `snow.3h`
  - `weather[].id`, `weather[].main`, `weather[].description`, `weather[].icon`

Uwagi z dokumentacji:

- Jeżeli nie widzisz niektórych pól w odpowiedzi, to znaczy, że zjawisko nie wystąpiło w danym czasie — API pokazuje tylko dane faktycznie zmierzone/obliczone.
- `main.temp_min` i `main.temp_max` w History API są opcjonalne i (w uproszczeniu) pokazują odchylenie temperatury w mieście w chwili obliczeń; w większości przypadków są równe `main.temp`. Nie mylić z dziennym `temp.min` / `temp.max` z dziennych prognoz (tam są to faktyczne min/max w ujęciu doby).

Przykład (history city):

```json
"main": {
  "temp": 306.15,
  "pressure": 1013,
  "humidity": 44,
  "temp_min": 306.15,
  "temp_max": 306.15
}
```

### Jednostki (`units`)

Parametr `units` jest opcjonalny: `standard`, `metric`, `imperial`. Jeśli nie ustawisz `units`, domyślnie używane jest `standard` (temperatura w kelwinach).

### Inne warianty wywołań (wbudowane geokodowanie / deprecated)

W dokumentacji są też warianty, które wołają historyczne dane po nazwie miasta (`q`) lub po identyfikatorze (`id`). Jest tam uwaga, że wbudowany geokoder i zapytania po nazwie/ID są wycofywane (mogą nadal działać, ale bez dalszych poprawek).

Po nazwie miasta:

`https://history.openweathermap.org/data/2.5/history/city?q={city_name},{country_code}&type=hour&start={start}&end={end}&appid={API_key}`

albo:

`https://history.openweathermap.org/data/2.5/history/city?q={city_name},{country_code}&type=hour&start={start}&cnt={cnt}&appid={API_key}`

Po ID miasta:

`https://history.openweathermap.org/data/2.5/history/city?id={id}&type=hour&start={start}&end={end}&appid={API_key}`

albo:

`https://history.openweathermap.org/data/2.5/history/city?id={id}&type=hour&start={start}&cnt={cnt}&appid={API_key}`

### Limit zakresu w jednej odpowiedzi

W dokumentacji jest uwaga, że dla planów Professional/Expert maksymalna „głębokość” danych w jednej odpowiedzi to 1 tydzień; jeśli poprosisz o dłuższy zakres, dostaniesz tylko pierwszy tydzień od `start` i trzeba robić kilka zapytań.
