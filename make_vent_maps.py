"""
=====================================================================================
 NBM VENTILATION RATE MAPS — Arizona / New Mexico PSAs, Days 1–7
=====================================================================================

WHAT THIS DOES
  Every morning GitHub runs this script. It:
    1. Loads the PSA (Predictive Service Area) boundaries            -> SECTION 2
    2. Finds the newest NBM model run and downloads ventilation rate -> SECTION 3
    3. Averages the afternoon hours for each of Days 1–7            -> SECTION 3
    4. Draws one overview map + one map per Southwest PSA           -> SECTIONS 4–6
    5. Writes the webpage (index.html)                              -> SECTION 7
  Everything is saved in the "site" folder, which GitHub publishes as the webpage.

RUN IT BY HAND (optional, on your own computer)
    pip install -r requirements.txt
    python make_vent_maps.py          then open site/index.html

-------------------------------------------------------------------------------------
 TROUBLESHOOTING — "something broke, where do I look?"
-------------------------------------------------------------------------------------
 Open the failed GitHub run > build-and-publish > "Make the maps and webpage" and find
 the last lines printed. Then match the message below:

 "Could not get PSA boundaries ..."
     The psa_boundaries.geojson file is missing from the repo AND NIFC's servers failed.
     Fix: upload psa_boundaries.geojson again.            Code: SECTION 2, fetch_psa_geojson()

 "PSAs mapped:" lists the wrong PSAs (or none)
     The PSA file's field names or GACC ID changed.
     Fix: check MAP_GACC_ID / INCLUDE_PSAS / EXCLUDE_PSAS. Code: SECTION 1 settings,
     SECTION 2 load_psas(). The log line "PSA attribute fields:" shows the field names.

 "No recent NBM cycle found on AWS or NOMADS."
     NOAA changed the NBM folder/file names, or both servers are down.
     Fix: check the address pattern.                      Code: SECTION 3, nbm_url()

 "fXXX: no VRATE in this file"  /  every day says "No data"
     NOAA renamed the ventilation variable inside the files.
     Fix: check the name 'VRATE'.                          Code: SECTION 3, vrate_byte_range()

 "Roads: could not load ..."
     The Natural Earth roads download failed. Maps still build, just without roads.
     Fix: usually goes away on the next run.               Code: SECTION 5, load_roads()

 Maps look wrong (colors, labels, crowding, zoom)
     Colors/categories -> SECTION 1 (CAT_...)
     Road markers      -> SECTION 1 (road settings) and SECTION 5 draw_roads()
     Overview layout   -> SECTION 6 make_overview()
     PSA map layout    -> SECTION 6 make_psa_map()

 Webpage text or look
     -> SECTION 7 (PAGE_CSS and write_index_html)

 The run never starts / wrong time
     -> not this file: see .github/workflows/vent-maps.yml (schedule times are UTC)
=====================================================================================
"""

# -------------------------------------------------------------------------------------
# Python packages. If GitHub says "ModuleNotFoundError: No module named X",
# add X to requirements.txt.
# -------------------------------------------------------------------------------------
import os
import re
import sys
import json
import html
import tempfile
import zipfile
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np                      # number crunching on the grid
import requests                         # downloads (NBM, NIFC)
import matplotlib
matplotlib.use('Agg')                   # draw to image files (no screen on GitHub)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs              # map projections
import cartopy.io.shapereader as shpreader   # Natural Earth downloads (states, roads)
from metpy.plots import USCOUNTIES, USSTATES  # county and state outlines
import pygrib                           # reads NBM GRIB2 files
import shapely                          # PSA polygon math
from shapely.geometry import shape
from shapely.ops import unary_union, linemerge


# =====================================================================================
# SECTION 1 — SETTINGS  (most day-to-day changes happen here)
# =====================================================================================

# ---- Output / webpage -----------------------------------------------------------------
OUTPUT_DIR = 'site'                     # folder GitHub publishes as the webpage
PAGE_TITLE = 'Ventilation Rate — Arizona & New Mexico PSAs'
# Address of the live webpage. The GitHub workflow fills this in; it's used to fetch
# yesterday's saved PSA boundaries as a last-resort backup.
SITE_URL = os.environ.get('SITE_URL', '').rstrip('/')

# ---- Time window ------------------------------------------------------------------------
DAYS = 7                                # number of forecast days (panels) per map
LOCAL_TZ = ZoneInfo('America/Phoenix')  # MST all year; decides what "today" (Day 1) is
# Time zone for the small "Issued" time in the legend panel (bottom right) of every map.
# America/Denver shows MDT in summer and MST in winter; use 'America/Phoenix' for MST all year.
ISSUED_TZ = ZoneInfo('America/Denver')
# Hours (UTC) used each day.
# 18 = 11 AM MST, 21 = 2 PM MST, 24 = 5 PM MST (midnight UTC of the next day).
AFTERNOON_UTC_HOURS = [18, 21, 24]
# How those hours are combined into one value per day:
#   'average' = mean of the hours (smoother, ignores one-hour spikes)   <- current
#   'max'     = highest of the hours
AFTERNOON_METHOD = 'average'
# Wording used on the maps and webpage for the choice above
#   legend = legend panel, title = map titles, page = webpage header/footnote
METHOD_TEXT = {
    'max':     {'legend': 'Daily max', 'title': 'Max Afternoon', 'page': 'daily max'},
    'average': {'legend': 'Afternoon', 'title': 'Afternoon', 'page': 'afternoon values'},
}

# ---- Which PSAs get maps ------------------------------------------------------------
# PSAs whose GACCUnitID matches this get their own map. Every other PSA nearby
# (Great Basin, Rocky Mtn, Southern Area, So Cal) is drawn only as a gray outline.
MAP_GACC_ID = 'USNMSWC'                 # Southwest Coordination Center
SHOW_NEIGHBOR_OUTLINES = True           # False = don't draw the neighboring PSAs at all
# Force a PSA in or out by its code, e.g. INCLUDE_PSAS = {'GB35'}, EXCLUDE_PSAS = {'SW03'}
INCLUDE_PSAS = set()
EXCLUDE_PSAS = set()
# Only used if a PSA file ever has NO GACC field: keep PSAs at least this much inside AZ/NM.
MIN_FRACTION_IN_AZ_NM = 0.5

# ---- Roads (Natural Earth 1:10m, downloaded automatically) -------------------------------
SHOW_ROADS = True                       # False = no roads anywhere
OVERVIEW_ROAD_LEVELS = []               # overview map: no roads (too cluttered)
#   to put interstates back on the overview: ['Interstate']
PSA_ROAD_LEVELS = ['Interstate', 'Federal']   # PSA maps: interstates + US highways
#   'Interstate' = I-10, I-25, I-40...   'Federal' = US 60, US 70, US 285...

# ---- Map area ----------------------------------------------------------------------------
# Area of NBM data that is downloaded/kept (a bit bigger than AZ/NM so edge PSAs and
# the bordering PSAs have data). Make this bigger if a map edge shows blank.
lat_min, lat_max = 30.0, 38.4
lon_min, lon_max = -116.8, -101.0
# What the overview map shows: [west, east, south, north]
OVERVIEW_EXTENT = [-115.9, -101.8, 30.7, 37.9]

# ---- Colors / categories (knot-ft), New Mexico AQB smoke management table -----------
# CAT_BOUNDS are the edges between categories; one more number than CAT_NAMES.
CAT_BOUNDS = [0, 40000, 60000, 100000, 150000, 10_000_000]
CAT_NAMES = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
CAT_COLORS = ['#7B3294', '#1F3A93', '#2E9E44', '#F2D300', '#D7191C']   # purple, blue, green, yellow, red
cmap = mcolors.ListedColormap(CAT_COLORS)
norm = mcolors.BoundaryNorm(CAT_BOUNDS, cmap.N)

# NBM gives ventilation in m²/s; this converts to knot-feet.
MPS_TO_KTFT = (1 / 0.514444) * (1 / 0.3048)

# ---- Spot check (troubleshooting) ----------------------------------------------------
# For each place below, the GitHub log prints the raw NBM value (m²/s) and the
# converted value (knot-ft) at every afternoon hour, so you can compare with other data.
# Add or remove places freely: 'Name': (latitude, longitude)
# Also log mixing height (ft) and the transport wind it implies (kt) at the check points.
# This only adds lines to the log; it doesn't change the maps. Costs ~1 extra minute per run.
LOG_MIXING_HEIGHT = True

CHECK_POINTS = {
    'Albuquerque': (35.0402, -106.6092),
    'Flagstaff': (35.1383, -111.6661),
    'Phoenix': (33.4342, -112.0116),
    'Roswell': (33.3016, -104.5306),
    'Silver City': (32.7701, -108.2803),
    'Socorro': (34.0584, -106.8914),
}

# ---- Data sources -------------------------------------------------------------------------
# NBM files are tried on AWS first, then NOAA's NOMADS server.
# If NOAA moves the files, update these and/or nbm_url() in SECTION 3.
NBM_SOURCES = [
    'https://noaa-nbm-grib2-pds.s3.amazonaws.com',
    'https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod',
]
# PSA boundaries: the copy uploaded to the repo is ALWAYS used first (no download).
# The NIFC addresses below are only a backup if that file is missing.
PSA_REPO_COPIES = ['psa_boundaries.geojson', 'psa_boundaries.zip']
PSA_SERVICES = [   # NIFC online services, newest first
    'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/'
    'DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer',
    'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/'
    'National_PSA_Current/FeatureServer',
]
PSA_GEOJSON_URLS = [
    'https://data-nifc.opendata.arcgis.com/datasets/nifc::predictive-service-area-psa-boundaries-public.geojson',
    'https://data-nifc.opendata.arcgis.com/datasets/nifc::national-predictive-service-areas-psa-boundaries.geojson',
]
# A small copy of the PSAs actually used is saved with the webpage each run.
PSA_CACHE = os.path.join(OUTPUT_DIR, 'psa_boundaries.geojson')


# =====================================================================================
# SECTION 2 — PSA BOUNDARIES
# Loads the PSA outlines and splits them into:
#   psas      = Southwest PSAs that get their own map
#   neighbors = nearby PSAs from other GACCs (gray outlines only)
# =====================================================================================

def read_repo_copy(path):
    """Read psa_boundaries.geojson, or the first .geojson file inside psa_boundaries.zip."""
    if path.lower().endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            inner = [n for n in z.namelist()
                     if n.lower().endswith(('.geojson', '.json')) and not n.startswith('__MACOSX')]
            if not inner:
                raise RuntimeError(f'{path} has no .geojson file inside')
            with z.open(inner[0]) as f:
                return json.load(f)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def fetch_psa_geojson():
    """Get the PSA boundaries. Order tried:
         1. psa_boundaries.geojson / .zip uploaded to the repo   <- normal case
         2. NIFC online services and downloads                    <- backup
         3. yesterday's copy saved on the webpage                 <- last resort
    """
    # 1. Repo copy (normal case)
    for path in PSA_REPO_COPIES:
        if os.path.exists(path):
            gj = read_repo_copy(path)
            print(f'PSA boundaries from the repo copy {path}: {len(gj.get("features", []))} features')
            return gj

    # 2. NIFC online (only if the repo copy is missing)
    params = {
        'where': '1=1', 'outFields': '*', 'outSR': '4326', 'f': 'geojson',
        'geometry': f'{lon_min},{lat_min},{lon_max},{lat_max}',
        'geometryType': 'esriGeometryEnvelope', 'inSR': '4326',
        'spatialRel': 'esriSpatialRelIntersects',
    }
    attempts = []
    for svc in PSA_SERVICES:
        for layer in (0, 1):
            url = f'{svc}/{layer}/query'
            attempts.append((url, lambda u=url: requests.get(u, params=params, timeout=60)))
    for url in PSA_GEOJSON_URLS:
        attempts.append((url, lambda u=url: requests.get(u, timeout=120)))
    # 3. Yesterday's copy on the live webpage
    if SITE_URL:
        url = f'{SITE_URL}/psa_boundaries.geojson'
        attempts.append((url, lambda u=url: requests.get(u, timeout=60)))

    for label, get in attempts:
        try:
            r = get()
            r.raise_for_status()
            gj = r.json()
            if gj.get('error'):                       # ArcGIS reports errors inside a "200 OK"
                raise RuntimeError(gj['error'])
            if gj.get('features'):
                print(f'PSA boundaries from {label}: {len(gj["features"])} features')
                return gj
            raise RuntimeError('no features in the response')
        except Exception as e:
            print(f'PSA boundaries: {label} failed ({str(e)[:200]})')

    if os.path.exists(PSA_CACHE):
        print(f'PSA boundaries: using the saved copy {PSA_CACHE}')
        with open(PSA_CACHE) as f:
            return json.load(f)
    raise RuntimeError('Could not get PSA boundaries from NIFC and no saved copy exists.')


# Field names the PSA code / name might be stored under. NIFC's current file uses
# PSANationalCode (e.g. "SW05") and PSANAME (e.g. "Western Mogollon Rim").
# If NIFC renames them, add the new name to these lists.
CODE_FIELDS = ['PSANationalCode', 'PSANATCODE', 'PSA_NAT_CODE', 'PSA_ID', 'PSAID',
               'PSACode', 'PSA_CODE', 'PSANumber', 'PSA_NUM', 'PSA']
NAME_FIELDS = ['PSAName', 'PSA_NAME', 'PSANAME', 'Name', 'NAME']
PSA_CODE_PATTERN = re.compile(r'^[A-Z]{2}\d{1,3}[A-Z]?$')   # what a PSA code looks like: SW05, SW06N


def first_value(props, keys):
    """Return the first non-empty value among these field names (not case-sensitive)."""
    for k in keys:
        for pk, v in props.items():
            if pk.lower() == k.lower() and v not in (None, ''):
                return str(v).strip()
    return ''


def guess_code(props):
    """Backup: if the code field has an unexpected name, find a value that looks like a PSA code."""
    for v in props.values():
        if isinstance(v, str) and PSA_CODE_PATTERN.match(v.strip()):
            return v.strip()
    return ''


def az_nm_shape():
    """AZ + NM outline (Natural Earth). Only needed if the PSA file has no GACC field."""
    path = shpreader.natural_earth(resolution='50m', category='cultural',
                                   name='admin_1_states_provinces_lakes')
    geoms = [rec.geometry for rec in shpreader.Reader(path).records()
             if rec.attributes.get('name') in ('Arizona', 'New Mexico')
             and rec.attributes.get('admin') == 'United States of America']
    return unary_union(geoms)


def load_psas():
    """Returns (psas, neighbors). See SECTION 1 'Which PSAs get maps' to change the choice."""
    gj = fetch_psa_geojson()
    # Printed so you can see the field names in the GitHub log if something changes.
    print('PSA attribute fields:', gj['features'][0].get('properties', {}))
    has_gacc = any(first_value(f.get('properties') or {}, ['GACCUnitID']) for f in gj['features'])
    azm = None if has_gacc else az_nm_shape()

    psas, neighbors = [], []
    for feat in gj['features']:
        props = feat.get('properties') or {}
        if not feat.get('geometry'):
            continue
        geom = shape(feat['geometry'])

        # Skip PSAs nowhere near AZ/NM (matters if the full national file is uploaded)
        bx0, by0, bx1, by1 = geom.bounds
        if bx1 < lon_min or bx0 > lon_max or by1 < lat_min or by0 > lat_max:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)                      # repairs small polygon errors

        code = first_value(props, CODE_FIELDS) or guess_code(props)
        name = first_value(props, NAME_FIELDS)
        if not code:
            continue

        # Southwest PSA (gets a map) or neighbor (outline only)?
        if has_gacc:
            in_area = first_value(props, ['GACCUnitID']) == MAP_GACC_ID
        else:
            in_area = (geom.intersection(azm).area / geom.area if geom.area else 0) >= MIN_FRACTION_IN_AZ_NM
        keep = (in_area or code in INCLUDE_PSAS) and code not in EXCLUDE_PSAS

        item = {'code': code, 'name': name, 'geom': geom, 'feature': feat,
                'file': f"vent_{re.sub(r'[^A-Za-z0-9]', '', code)}.png"}   # image file name, e.g. vent_SW05.png
        (psas if keep else neighbors).append(item)

    psas.sort(key=lambda p: p['code'])
    if not SHOW_NEIGHBOR_OUTLINES:
        neighbors = []

    # Save the PSAs used here with the webpage (backup copy for a future run)
    with open(PSA_CACHE, 'w') as f:
        json.dump({'type': 'FeatureCollection',
                   'features': [p['feature'] for p in psas + neighbors]}, f)
    print('PSAs mapped:', ', '.join(f"{p['code']} ({p['name']})" for p in psas))
    print('Neighbor outlines:', ', '.join(sorted(p['code'] for p in neighbors)) or 'none')
    return psas, neighbors


# =====================================================================================
# SECTION 3 — NBM DATA
# NBM files are large, so only the ventilation (VRATE) piece of each file is downloaded,
# using the small ".idx" index file that lists where each variable sits in the file.
# =====================================================================================

def nbm_url(base, cycle, fhr):
    """Address of one NBM CONUS 'core' file, e.g.
       .../blend.20261027/12/core/blend.t12z.core.f030.co.grib2
       If NOAA changes its folder or file naming, THIS is the line to fix."""
    return (f'{base}/blend.{cycle:%Y%m%d}/{cycle:%H}/core/'
            f'blend.t{cycle:%H}z.core.f{fhr:03d}.co.grib2')


def get_idx(cycle, fhr):
    """Download the .idx index for one forecast hour. Tries each server in NBM_SOURCES.
       Returns (grib_url, idx_text), or (None, None) if no server has it yet."""
    for base in NBM_SOURCES:
        url = nbm_url(base, cycle, fhr)
        try:
            r = requests.get(url + '.idx', timeout=30)
            if r.status_code == 200 and r.text.strip():
                return url, r.text
        except requests.RequestException:
            pass
    return None, None


def vrate_byte_range(idx_text, var='VRATE'):
    """Find the ventilation rate (VRATE) entry in the .idx file -> (start byte, end byte).
       An .idx line looks like:  123:45678901:d=2026102712:VRATE:entire atmosphere ...:6 hour fcst:
       If NOAA renames the variable, change 'VRATE' here."""
    lines = [l for l in idx_text.strip().splitlines() if l.strip()]
    for i, line in enumerate(lines):
        parts = line.split(':')
        if len(parts) > 4 and parts[3] == var and 'std dev' not in line and 'prob' not in line:
            start = int(parts[1])
            # The piece ends where the next entry starts (or at the end of the file)
            end = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else None
            return start, end
    return None


def valid_times_for_day(day):
    """The afternoon times (UTC) checked for one day; see AFTERNOON_UTC_HOURS in SECTION 1."""
    base = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    return [base + dt.timedelta(hours=h) for h in AFTERNOON_UTC_HOURS]


def fhr_for(cycle, valid):
    """Forecast hour = hours from the model run time to the valid time."""
    return int((valid - cycle).total_seconds() // 3600)


def pick_cycle(days):
    """Choose the newest NBM run that already has its Day 7 file posted.
       Tries today's 18Z, 12Z, 06Z, 00Z, then yesterday's 18Z, 12Z.
       Runs less than 45 minutes old are skipped (not posted yet)."""
    now = dt.datetime.now(dt.timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = sorted(
        [today + dt.timedelta(hours=h) for h in (18, 12, 6, 0)] +
        [today - dt.timedelta(days=1) + dt.timedelta(hours=h) for h in (18, 12)],
        reverse=True)
    candidates = [c for c in candidates if c <= now - dt.timedelta(minutes=45)]

    last_valid = valid_times_for_day(days[-1])[-1]
    first_valid = valid_times_for_day(days[0])[0]
    for c in candidates:                       # best: a run that reaches Day 7
        if get_idx(c, fhr_for(c, last_valid))[0]:
            return c
    for c in candidates:                       # fallback: a run with at least Day 1
        f = fhr_for(c, first_valid)
        if f >= 1 and get_idx(c, f)[0]:
            return c
    raise RuntimeError('No recent NBM cycle found on AWS or NOMADS.')


class Grid:
    """Latitude/longitude of the NBM grid points kept (AZ/NM area). Filled on first read."""
    lats = lons = sl = None


def read_vrate(cycle, fhr):
    """Ventilation rate grid in knot-ft (AZ/NM area only), or None if not available."""
    vals = read_field(cycle, fhr, 'VRATE')
    return None if vals is None else vals * MPS_TO_KTFT       # m²/s -> knot-ft


def read_field(cycle, fhr, var, quiet=False):
    """Download just one variable (e.g. 'VRATE' or 'MIXHT') for one forecast hour.
       Returns the grid in the file's own units (AZ/NM area only), or None if not available."""
    if fhr < 1 or fhr > 264:                   # NBM core files run f001–f264
        return None
    url, idx = get_idx(cycle, fhr)
    if not url:
        print(f'  f{fhr:03d}: not posted')
        return None
    rng = vrate_byte_range(idx, var)
    if not rng:
        print(f'  f{fhr:03d}: no {var} in this file')
        return None

    # Download only the bytes for this variable (a "Range" request)
    start, end = rng
    headers = {'Range': f'bytes={start}-{end if end is not None else ""}'}
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()

    # pygrib needs a file on disk, so save the piece to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name
    try:
        grbs = pygrib.open(path)
        g = grbs.message(1)
        # Log line showing exactly what was read: variable name, UNITS, forecast hour.
        # The units should say m**2 s**-1 (m²/s). If they ever say something else,
        # MPS_TO_KTFT in SECTION 1 is the wrong conversion.
        if not quiet:
            print(f'  f{fhr:03d}: {g}')
        vals = np.ma.filled(np.ma.asarray(g.values, dtype=float), np.nan)
        if Grid.sl is None:
            # First file only: work out which grid rows/columns cover the AZ/NM area
            lats, lons = g.latlons()
            lons = np.where(lons > 180, lons - 360, lons)
            inside = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
            rows = np.where(inside.any(axis=1))[0]
            cols = np.where(inside.any(axis=0))[0]
            Grid.sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
            Grid.lats, Grid.lons = lats[Grid.sl], lons[Grid.sl]
        grbs.close()
    finally:
        os.remove(path)

    vals = vals[Grid.sl]
    vals[vals > 1e6] = np.nan                  # NBM marks missing points with huge numbers
    return vals


def nearest_index(lat, lon):
    """Grid row/column closest to a lat/lon (for the CHECK_POINTS spot check)."""
    d = (Grid.lats - lat) ** 2 + ((Grid.lons - lon) * np.cos(np.radians(lat))) ** 2
    return np.unravel_index(np.nanargmin(d), d.shape)


def afternoon_value(cycle, day):
    """One value per grid point for the day: the average (or max) of the afternoon hours,
       set by AFTERNOON_METHOD in SECTION 1.
       Also prints the CHECK_POINTS values to the log for troubleshooting."""
    fields = []
    for valid in valid_times_for_day(day):
        f = fhr_for(cycle, valid)
        v = read_vrate(cycle, f)
        if v is None:
            continue
        fields.append(v)
        local = valid.astimezone(LOCAL_TZ)
        # Mixing height (log only - does not change the map). Transport wind isn't in the
        # gridded NBM, so it is worked out as ventilation / mixing height.
        mix = read_field(cycle, f, 'MIXHT', quiet=True) if LOG_MIXING_HEIGHT else None
        for name, (lat, lon) in CHECK_POINTS.items():
            iy, ix = nearest_index(lat, lon)
            ktft = v[iy, ix]
            line = (f'    {name:12} {local:%a %m/%d %I %p} MST (f{f:03d}): '
                    f'{ktft / MPS_TO_KTFT:8.0f} m²/s = {ktft:9,.0f} kt-ft  {category(ktft)}')
            if mix is not None and mix[iy, ix] > 0:
                mh_m = mix[iy, ix]
                tw_kt = (ktft / MPS_TO_KTFT) / mh_m / 0.514444      # (m²/s ÷ m) = m/s -> knots
                line += f' | mix hgt {mh_m / 0.3048:7,.0f} ft | transport wind {tw_kt:4.0f} kt (implied)'
            print(line)
    if not fields:
        return None                            # shows as "No data" on the map
    with np.errstate(all='ignore'):
        if AFTERNOON_METHOD == 'max':
            combined = np.fmax.reduce(fields)
        else:
            combined = np.nanmean(np.stack(fields), axis=0)
    # Log the combined value at each check point (this is what the map shows there)
    for name, (lat, lon) in CHECK_POINTS.items():
        iy, ix = nearest_index(lat, lon)
        val = combined[iy, ix]
        print(f'    {name:12} {METHOD_TEXT[AFTERNOON_METHOD]["legend"].upper():>24}: '
              f'{val:9,.0f} kt-ft  {category(val)}   <- shown on map')
    return combined


# =====================================================================================
# SECTION 4 — SHARED DRAWING PIECES (legend, colors, category names)
# =====================================================================================

def category(v):
    """Knot-ft value -> category name (Poor ... Excellent), using CAT_BOUNDS."""
    if v is None or np.isnan(v):
        return '—'
    for i in range(len(CAT_NAMES)):
        if v < CAT_BOUNDS[i + 1]:
            return CAT_NAMES[i]
    return CAT_NAMES[-1]


def day_label(day):
    """Date as shown on the maps, e.g. 'Tue 10/27'."""
    return f'{day:%a} {day.month}/{day.day}'


def legend_panel(ax, cycle):
    """The 8th (bottom-right) panel: color legend, issued time, model run."""
    ax.axis('off')
    handles = [Patch(facecolor=c, edgecolor='black',
                     label=f'{n}  ({CAT_BOUNDS[i] // 1000}K–{CAT_BOUNDS[i + 1] // 1000}K)'
                     if i < len(CAT_NAMES) - 1 else f'{n}  (≥{CAT_BOUNDS[i] // 1000}K)')
               for i, (n, c) in enumerate(zip(CAT_NAMES, CAT_COLORS))]
    ax.legend(handles=handles, loc='upper center', title='Ventilation (knot-ft)',
              fontsize=10, title_fontsize=11, frameon=False)
    # Small gray "Issued" time (when GitHub ran the script), just above the model run
    ax.text(0.5, 0.24, issued_text(), ha='center', va='center',
            fontsize=8, color='#555555', transform=ax.transAxes)
    ax.text(0.5, 0.17, f'Model run: {cycle:%H}Z {cycle.month}/{cycle.day}/{cycle.year}',
            ha='center', va='center', fontsize=9, transform=ax.transAxes)


ISSUED = dt.datetime.now(dt.timezone.utc)   # set once when the run starts (see main())


def issued_text():
    """'Issued ...' line shown in the legend panel (bottom right), above the model run.
       Size/color are set in legend_panel(); time zone is ISSUED_TZ in SECTION 1."""
    t = ISSUED.astimezone(ISSUED_TZ)
    hour = t.strftime('%I').lstrip('0')            # 7 instead of 07
    return f'Issued {hour}:{t:%M %p %Z} {t:%a} {t.month}/{t.day}/{t.year}'


def draw_field(ax, data):
    """The colored ventilation grid for one day (or 'No data')."""
    if data is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, transform=ax.transAxes)
        return
    ax.pcolormesh(Grid.lons, Grid.lats, data, transform=ccrs.PlateCarree(),
                  cmap=cmap, norm=norm, shading='nearest', zorder=1)


def psa_mask(geom):
    """True for grid points inside a PSA (used for the PSA median)."""
    return shapely.contains_xy(geom, Grid.lons, Grid.lats)


# =====================================================================================
# SECTION 5 — ROADS AND NEIGHBOR OUTLINES
# Roads come from Natural Earth (downloaded automatically, then cached).
# Which roads show on which map is set in SECTION 1 (OVERVIEW_ROAD_LEVELS, PSA_ROAD_LEVELS).
# =====================================================================================

ROADS = None   # filled the first time load_roads() runs, then reused for every map


def load_roads():
    """Interstates and US highways near AZ/NM -> {(level, route number): line geometry}.
       If the download fails, maps are still drawn, just without roads."""
    global ROADS
    if ROADS is not None:
        return ROADS
    ROADS = {}
    if not SHOW_ROADS:
        return ROADS
    try:
        path = shpreader.natural_earth(resolution='10m', category='cultural', name='roads')
        area = shapely.box(lon_min, lat_min, lon_max, lat_max)
        parts, seen_levels = {}, set()
        for rec in shpreader.Reader(path).records():
            a = rec.attributes
            # Natural Earth stores the road class in "level" (Interstate / Federal / State ...)
            raw = str(a.get('level') or a.get('level_') or '').strip()
            if a.get('sov_a3') not in (None, 'USA'):   # US roads only
                continue
            seen_levels.add(raw)
            low = raw.lower()
            if 'interstate' in low:
                level = 'Interstate'
            elif 'federal' in low or low in ('us', 'u.s.'):
                level = 'Federal'
            else:
                continue                             # state/local roads are skipped
            geom = rec.geometry
            if geom is None or not geom.intersects(area):
                continue
            # Route number for the marker, e.g. "40" from "I-40" or "US 60"
            num = ''
            for field in ('label', 'name', 'routeraw'):
                m = re.search(r'\d+', str(a.get(field) or ''))
                if m:
                    num = m.group(0)
                    break
            parts.setdefault((level, num), []).append(geom.intersection(area))
        ROADS = {k: unary_union(v) for k, v in parts.items()}
        print('Road levels in the file:', sorted(seen_levels))
        print(f'Roads: {sum(1 for k in ROADS if k[0] == "Interstate")} interstates, '
              f'{sum(1 for k in ROADS if k[0] == "Federal")} US highways')
    except Exception as e:
        print(f'Roads: could not load ({e}) - maps will be drawn without roads')
    return ROADS


def longest_line(geom):
    """Longest continuous stretch of a road (where its marker goes)."""
    if geom.is_empty:
        return None
    if geom.geom_type == 'MultiLineString':
        geom = linemerge(geom)
    lines = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
    lines = [g for g in lines if g.geom_type == 'LineString']
    return max(lines, key=lambda g: g.length) if lines else None


def draw_roads(ax, extent, levels, fontsize):
    """Draw road lines + one route marker per road on one map panel.
       levels = which road classes to draw, e.g. ['Interstate', 'Federal'] (empty = none)."""
    if not levels:
        return
    roads = load_roads()
    if not roads:
        return
    x0, x1, y0, y1 = extent
    view = shapely.box(x0, y0, x1, y1)

    # Line look (color/thickness) for each road class.
    # Use "edgecolor", NOT "color": "color" also fills the shape and draws dark wedges.
    style = {'Interstate': dict(edgecolor='#222222', linewidth=1.1),
             'Federal': dict(edgecolor='#444444', linewidth=0.6)}
    # Marker look: interstates = blue rounded box, US highways = white square box
    shield = {'Interstate': dict(facecolor='#1f4e9c', edgecolor='white', boxstyle='round,pad=0.25'),
              'Federal': dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2')}
    # No marker on road pieces shorter than this (fraction of the map width); raise to reduce clutter
    min_len = 0.08 * max(x1 - x0, y1 - y0)

    for level in reversed(levels):              # US highways first, interstates drawn on top
        for (lvl, num), geom in roads.items():
            if lvl != level:
                continue
            visible = geom.intersection(view)
            if visible.is_empty:
                continue
            ax.add_geometries([visible], ccrs.PlateCarree(), facecolor='none',
                              alpha=0.85, zorder=3.5, **style[level])
            line = longest_line(visible)
            if num and line is not None and line.length >= min_len:
                pt = line.interpolate(0.5, normalized=True)       # middle of the longest stretch
                ax.text(pt.x, pt.y, num, transform=ccrs.PlateCarree(), fontsize=fontsize,
                        fontweight='bold', ha='center', va='center', zorder=6, clip_on=True,
                        color='white' if level == 'Interstate' else 'black', bbox=shield[level])


def draw_neighbors(ax, neighbors, extent, fontsize):
    """Bordering PSAs from other GACCs: gray dashed outline + gray italic code (context only)."""
    if not neighbors:
        return
    ax.add_geometries([p['geom'] for p in neighbors], ccrs.PlateCarree(), facecolor='none',
                      edgecolor='#555555', linewidth=0.7, linestyle='--', zorder=3)
    x0, x1, y0, y1 = extent
    for p in neighbors:
        visible = p['geom'].intersection(shapely.box(x0, y0, x1, y1))
        if visible.is_empty or visible.area < 0.02:   # skip labels for slivers at the map edge
            continue
        pt = visible.representative_point()
        ax.text(pt.x, pt.y, p['code'], transform=ccrs.PlateCarree(), fontsize=fontsize,
                color='#444444', fontstyle='italic', ha='center', va='center', zorder=5, clip_on=True,
                bbox=dict(facecolor='white', alpha=0.55, edgecolor='none', pad=0.4))


# =====================================================================================
# SECTION 6 — THE MAPS
# Each image is 16x9 inches at 120 dpi (1920x1080): 7 day panels + 1 legend panel,
# laid out 2 rows x 4 columns.
# =====================================================================================

def make_overview(psas, neighbors, days, fields, cycle):
    """vent_overview.png — all Southwest PSAs, Days 1–7. No roads (see SECTION 1)."""
    fig = plt.figure(figsize=(16, 9))
    extent = OVERVIEW_EXTENT
    for i, (day, data) in enumerate(zip(days, fields)):
        ax = fig.add_subplot(2, 4, i + 1, projection=ccrs.Mercator())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        draw_field(ax, data)                                               # colors
        ax.add_feature(USSTATES.with_scale('5m'), edgecolor='black', linewidth=1.0, zorder=3)  # states
        draw_neighbors(ax, neighbors, extent, fontsize=5)                  # gray neighbor PSAs
        draw_roads(ax, extent, OVERVIEW_ROAD_LEVELS, fontsize=4.5)         # roads (off by default)
        ax.add_geometries([p['geom'] for p in psas], ccrs.PlateCarree(),   # SW PSA outlines
                          facecolor='none', edgecolor='black', linewidth=0.6, zorder=4)
        for p in psas:                                                     # SW PSA code labels
            pt = p['geom'].representative_point()
            ax.text(pt.x, pt.y, p['code'], transform=ccrs.PlateCarree(), fontsize=5.5,
                    fontweight='bold', ha='center', va='center', zorder=5,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
        ax.set_title(f'Day {i + 1}: {day_label(day)}', fontsize=11, fontweight='bold')
    legend_panel(fig.add_subplot(2, 4, 8), cycle)
    fig.suptitle(f'{METHOD_TEXT[AFTERNOON_METHOD]["title"]} Ventilation Rate — Arizona & New Mexico PSAs',
                 fontsize=16, fontweight='bold')
    # Spacing between panels. hspace = gap between the two rows (raise it if a
    # day title touches the map above); top = room for the big title.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90, wspace=0.06, hspace=0.18)
    out = os.path.join(OUTPUT_DIR, 'vent_overview.png')
    fig.savefig(out, dpi=120, facecolor='white')
    plt.close(fig)
    print('saved', out)


def make_psa_map(psa, psas, neighbors, days, fields, cycle):
    """vent_<PSA>.png — one PSA zoomed in, Days 1–7, with counties, roads and the PSA median."""
    # Zoom: the PSA's bounding box plus a margin (raise 0.15 to zoom out more)
    minx, miny, maxx, maxy = psa['geom'].bounds
    pad = 0.15 * max(maxx - minx, maxy - miny) + 0.1
    extent = [minx - pad, maxx + pad, miny - pad, maxy + pad]
    mask = psa_mask(psa['geom'])
    others = [p['geom'] for p in psas if p is not psa]

    fig = plt.figure(figsize=(16, 9))
    for i, (day, data) in enumerate(zip(days, fields)):
        ax = fig.add_subplot(2, 4, i + 1, projection=ccrs.Mercator())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        draw_field(ax, data)                                                                    # colors
        ax.add_feature(USCOUNTIES.with_scale('500k'), edgecolor='gray', linewidth=0.4, zorder=2)  # counties
        ax.add_feature(USSTATES.with_scale('500k'), edgecolor='black', linewidth=1.0, zorder=3)   # states
        draw_neighbors(ax, neighbors, extent, fontsize=7)                                       # gray neighbor PSAs
        draw_roads(ax, extent, PSA_ROAD_LEVELS, fontsize=6.5)                                   # roads + markers
        if others:                                                                              # other SW PSAs
            ax.add_geometries(others, ccrs.PlateCarree(), facecolor='none',
                              edgecolor='dimgray', linewidth=0.8, linestyle='--', zorder=3)
        ax.add_geometries([psa['geom']], ccrs.PlateCarree(), facecolor='none',                  # this PSA (bold)
                          edgecolor='black', linewidth=2.2, zorder=4)

        # Panel title: date + median ventilation inside the PSA and its category
        if data is not None and mask.any():
            with np.errstate(all='ignore'):
                med = float(np.nanmedian(data[mask]))
            sub = f'Median {med:,.0f} kt-ft · {category(med)}' if not np.isnan(med) else 'No data'
        else:
            sub = 'No data'
        ax.set_title(f'Day {i + 1}: {day_label(day)}\n{sub}', fontsize=10, fontweight='bold')

    legend_panel(fig.add_subplot(2, 4, 8), cycle)
    title = f"{psa['code']} {psa['name']}".strip()
    fig.suptitle(f'{title} — {METHOD_TEXT[AFTERNOON_METHOD]["title"]} Ventilation Rate', fontsize=16, fontweight='bold')
    # Spacing between panels. hspace = gap between the two rows; these titles are two
    # lines, so the gap is bigger. Raise hspace if a title touches the map above.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88, wspace=0.06, hspace=0.34)
    out = os.path.join(OUTPUT_DIR, psa['file'])
    fig.savefig(out, dpi=120, facecolor='white')
    plt.close(fig)
    print('saved', out)


# =====================================================================================
# SECTION 7 — THE WEBPAGE (index.html)
# PAGE_CSS controls the look (colors, spacing). write_index_html() controls the content.
# =====================================================================================

PAGE_CSS = """
:root { --bg:#f5f6f8; --card:#ffffff; --text:#1d2330; --muted:#5b6475; --line:#d9dde4; --accent:#1f3a93; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#12151b; --card:#1b2029; --text:#e6e9ef; --muted:#9aa3b2; --line:#2c3340; --accent:#8fb0ff; }
}
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
       background:var(--bg); color:var(--text); line-height:1.4; }
header { padding:20px 16px 8px; max-width:1400px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 4px; }
.meta { color:var(--muted); font-size:0.95rem; }
nav { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
      padding:8px 16px; }
nav .inner { max-width:1400px; margin:0 auto; display:flex; flex-wrap:wrap; gap:6px; }
nav a { text-decoration:none; color:var(--accent); border:1px solid var(--line); background:var(--card);
        border-radius:6px; padding:4px 10px; font-size:0.9rem; font-weight:600; }
nav a:hover { border-color:var(--accent); }
main { max-width:1400px; margin:0 auto; padding:8px 16px 40px; }
section { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:12px; margin:16px 0; scroll-margin-top:60px; }
section h2 { font-size:1.15rem; margin:0 0 8px; }
section h2 span { color:var(--muted); font-weight:500; }
img { width:100%; height:auto; display:block; border-radius:6px; background:#fff; }
.note { color:var(--muted); font-size:0.85rem; margin-top:24px; }
"""


def write_index_html(psas, days, cycle):
    """Build the webpage: header, PSA buttons, overview image, one section per PSA, footnote."""
    stamp = f'{cycle:%Y%m%d%H}'      # added to image links so browsers load today's images, not cached ones
    built = dt.datetime.now(LOCAL_TZ)
    esc = html.escape

    # Row of buttons at the top (Overview, SW01, SW02, ...)
    nav = ['<a href="#overview">Overview</a>'] + [
        f'<a href="#{esc(re.sub(r"[^A-Za-z0-9]", "", p["code"]))}" title="{esc(p["name"])}">{esc(p["code"])}</a>'
        for p in psas]

    # One box per image
    sections = [
        '<section id="overview"><h2>All AZ/NM PSAs</h2>'
        f'<img src="vent_overview.png?v={stamp}" alt="Ventilation overview, Days 1 to 7"></section>'
    ]
    for p in psas:
        anchor = re.sub(r'[^A-Za-z0-9]', '', p['code'])
        sections.append(
            f'<section id="{esc(anchor)}"><h2>{esc(p["code"])} <span>{esc(p["name"])}</span></h2>'
            f'<img src="{esc(p["file"])}?v={stamp}" loading="lazy" '
            f'alt="{esc(p["code"])} ventilation, Days 1 to 7"></section>')

    # The page itself. The footnote text at the bottom can be edited freely.
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(PAGE_TITLE)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header>
  <h1>{esc(PAGE_TITLE)}</h1>
  <div class="meta">Days 1–7 ({day_label(days[0])} – {day_label(days[-1])}) ·
    model run {cycle:%H}Z {cycle.month}/{cycle.day}/{cycle.year} ·
    {METHOD_TEXT[AFTERNOON_METHOD]['page']} · updated {built:%-I:%M %p} MST {built.month}/{built.day}</div>
</header>
<nav><div class="inner">{''.join(nav)}</div></nav>
<main>
{chr(10).join(sections)}
<p class="note">Ventilation rate = mixing height × transport wind, from the National Blend of Models,
converted to knot-feet. Categories follow the New Mexico smoke management table:
Poor &lt;40K, Fair 40–60K, Good 60–100K, Very Good 100–150K, Excellent ≥150K.
PSA boundaries: NIFC. Each PSA panel shows the median value inside the PSA.</p>
</main>
</body>
</html>
"""
    with open(os.path.join(OUTPUT_DIR, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(page)
    print('saved index.html')


# =====================================================================================
# SECTION 8 — MAIN: runs everything in order
# =====================================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: PSA boundaries (SECTION 2)
    psas, neighbors = load_psas()
    if not psas:
        sys.exit('No AZ/NM PSAs found.')

    # Step 2: pick the NBM run (SECTION 3). Day 1 = today in Arizona time.
    today = dt.datetime.now(LOCAL_TZ).date()
    days = [today + dt.timedelta(days=i) for i in range(DAYS)]
    global ISSUED
    ISSUED = dt.datetime.now(dt.timezone.utc)      # "Issued" time stamped on every map
    cycle = pick_cycle(days)
    print(f'Using NBM {cycle:%Y-%m-%d %H}Z')

    # Step 3: afternoon ventilation (average or max) for each day (SECTION 3)
    fields = []
    for day in days:
        print(f'Day {day}:')
        fields.append(afternoon_value(cycle, day))
    if all(f is None for f in fields):
        sys.exit('No NBM ventilation data could be read.')

    # Step 4: maps (SECTION 6)
    make_overview(psas, neighbors, days, fields, cycle)
    for psa in psas:
        make_psa_map(psa, psas, neighbors, days, fields, cycle)

    # Step 5: webpage (SECTION 7)
    write_index_html(psas, days, cycle)


if __name__ == '__main__':
    main()
