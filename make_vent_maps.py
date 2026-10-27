"""
NBM max afternoon ventilation rate maps for the Arizona / New Mexico
Predictive Service Areas (PSAs), Days 1-7.

Builds a small website in site/:
  index.html               - the webpage (overview + one section per PSA)
  vent_overview.png        - all AZ/NM PSAs, Days 1-7
  vent_<PSA>.png           - one image per PSA, Days 1-7
  psa_boundaries.geojson   - copy of the PSA boundaries (backup if NIFC is down)

Run daily by GitHub Actions and published with GitHub Pages
(see .github/workflows/vent-maps.yml). Can also be run by hand:
  python make_vent_maps.py      then open site/index.html
"""
import os
import re
import sys
import json
import html
import tempfile
import zipfile
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from metpy.plots import USCOUNTIES, USSTATES
import pygrib
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union, linemerge

# ---- Settings ----------------------------------------------------------------
OUTPUT_DIR = 'site'
PAGE_TITLE = 'NBM Ventilation Rate — Arizona & New Mexico PSAs'
# The live site (set by the GitHub workflow); its saved PSA boundaries are the backup copy.
SITE_URL = os.environ.get('SITE_URL', '').rstrip('/')
DAYS = 7
LOCAL_TZ = ZoneInfo('America/Phoenix')       # MST all year
AFTERNOON_UTC_HOURS = [18, 21, 24]           # 11 AM, 2 PM, 5 PM MST -> daily max of these
AFTERNOON_TEXT = '11 AM, 2 PM, 5 PM MST'
MIN_FRACTION_IN_AZ_NM = 0.5                  # (only used if the file has no GACC field)

# Which PSAs get their own maps: those in this GACC. Every other PSA near AZ/NM
# (Great Basin, Rocky Mtn, Southern Area, So Cal) is drawn as a gray outline for context.
MAP_GACC_ID = 'USNMSWC'                      # Southwest Coordination Center
SHOW_NEIGHBOR_OUTLINES = True

# Roads (Natural Earth 1:10m, downloaded automatically)
SHOW_ROADS = True
OVERVIEW_ROAD_LEVELS = ['Interstate']                 # overview panels are small: interstates only
PSA_ROAD_LEVELS = ['Interstate', 'Federal']           # PSA maps: interstates + US highways

# PSA codes to always skip or always include.
EXCLUDE_PSAS = set()
INCLUDE_PSAS = set()

# Data area (bigger than AZ/NM so edge PSAs and the bordering PSAs are covered)
lat_min, lat_max = 30.0, 38.4
lon_min, lon_max = -116.8, -101.0
# What the overview map shows (AZ/NM plus a strip of the neighboring PSAs)
OVERVIEW_EXTENT = [-115.9, -101.8, 30.7, 37.9]

# Smoke management categories (knot-ft), New Mexico AQB table
CAT_BOUNDS = [0, 40000, 60000, 100000, 150000, 10_000_000]
CAT_NAMES = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
CAT_COLORS = ['#7B3294', '#1F3A93', '#2E9E44', '#F2D300', '#D7191C']
cmap = mcolors.ListedColormap(CAT_COLORS)
norm = mcolors.BoundaryNorm(CAT_BOUNDS, cmap.N)

MPS_TO_KTFT = (1 / 0.514444) * (1 / 0.3048)    # m^2/s -> knot-feet

NBM_SOURCES = [
    'https://noaa-nbm-grib2-pds.s3.amazonaws.com',
    'https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod',
]
PSA_SERVICES = [   # NIFC PSA boundary services, newest first
    'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/'
    'DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer',
    'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/'
    'National_PSA_Current/FeatureServer',
]
PSA_GEOJSON_URLS = [
    'https://data-nifc.opendata.arcgis.com/datasets/nifc::predictive-service-area-psa-boundaries-public.geojson',
    'https://data-nifc.opendata.arcgis.com/datasets/nifc::national-predictive-service-areas-psa-boundaries.geojson',
]
PSA_CACHE = os.path.join(OUTPUT_DIR, 'psa_boundaries.geojson')
# A copy uploaded to the repo itself, plain or zipped. If one exists, it is always used (no download).
PSA_REPO_COPIES = ['psa_boundaries.geojson', 'psa_boundaries.zip']


# ---- PSA boundaries ------------------------------------------------------------
def read_repo_copy(path):
    """Read psa_boundaries.geojson, or the first .geojson/.json file inside psa_boundaries.zip."""
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
    # A copy uploaded to the repo always wins, so NIFC's servers aren't needed at all.
    for path in PSA_REPO_COPIES:
        if os.path.exists(path):
            gj = read_repo_copy(path)
            print(f'PSA boundaries from the repo copy {path}: {len(gj.get("features", []))} features')
            return gj

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
    if SITE_URL:
        url = f'{SITE_URL}/psa_boundaries.geojson'
        attempts.append((url, lambda u=url: requests.get(u, timeout=60)))

    for label, get in attempts:
        try:
            r = get()
            r.raise_for_status()
            gj = r.json()
            if gj.get('error'):
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


CODE_FIELDS = ['PSANationalCode', 'PSANATCODE', 'PSA_NAT_CODE', 'PSA_ID', 'PSAID',
               'PSACode', 'PSA_CODE', 'PSANumber', 'PSA_NUM', 'PSA']
NAME_FIELDS = ['PSAName', 'PSA_NAME', 'PSANAME', 'Name', 'NAME']
PSA_CODE_PATTERN = re.compile(r'^[A-Z]{2}\d{1,3}[A-Z]?$')   # e.g. SW05, SW06N


def first_value(props, keys):
    for k in keys:
        for pk, v in props.items():
            if pk.lower() == k.lower() and v not in (None, ''):
                return str(v).strip()
    return ''


def guess_code(props):
    """If the code field has an unexpected name, find a value that looks like a PSA code."""
    for v in props.values():
        if isinstance(v, str) and PSA_CODE_PATTERN.match(v.strip()):
            return v.strip()
    return ''


def az_nm_shape():
    path = shpreader.natural_earth(resolution='50m', category='cultural',
                                   name='admin_1_states_provinces_lakes')
    geoms = [rec.geometry for rec in shpreader.Reader(path).records()
             if rec.attributes.get('name') in ('Arizona', 'New Mexico')
             and rec.attributes.get('admin') == 'United States of America']
    return unary_union(geoms)


def load_psas():
    """Returns (psas, neighbors): PSAs that get maps, and nearby PSAs drawn as outlines only."""
    gj = fetch_psa_geojson()
    print('PSA attribute fields:', gj['features'][0].get('properties', {}))
    has_gacc = any(first_value(f.get('properties') or {}, ['GACCUnitID']) for f in gj['features'])
    azm = None if has_gacc else az_nm_shape()

    psas, neighbors = [], []
    for feat in gj['features']:
        props = feat.get('properties') or {}
        if not feat.get('geometry'):
            continue
        geom = shape(feat['geometry'])
        bx0, by0, bx1, by1 = geom.bounds
        if bx1 < lon_min or bx0 > lon_max or by1 < lat_min or by0 > lat_max:
            continue                            # nowhere near AZ/NM (quick skip for the national file)
        if not geom.is_valid:
            geom = geom.buffer(0)
        code = first_value(props, CODE_FIELDS) or guess_code(props)
        name = first_value(props, NAME_FIELDS)
        if not code:
            continue
        if has_gacc:
            in_area = first_value(props, ['GACCUnitID']) == MAP_GACC_ID
        else:
            in_area = (geom.intersection(azm).area / geom.area if geom.area else 0) >= MIN_FRACTION_IN_AZ_NM
        keep = (in_area or code in INCLUDE_PSAS) and code not in EXCLUDE_PSAS
        item = {'code': code, 'name': name, 'geom': geom, 'feature': feat,
                'file': f"vent_{re.sub(r'[^A-Za-z0-9]', '', code)}.png"}
        (psas if keep else neighbors).append(item)
    psas.sort(key=lambda p: p['code'])
    if not SHOW_NEIGHBOR_OUTLINES:
        neighbors = []

    # Save the PSAs used here as a small backup copy on the website
    with open(PSA_CACHE, 'w') as f:
        json.dump({'type': 'FeatureCollection',
                   'features': [p['feature'] for p in psas + neighbors]}, f)
    print('PSAs mapped:', ', '.join(f"{p['code']} ({p['name']})" for p in psas))
    print('Neighbor outlines:', ', '.join(sorted(p['code'] for p in neighbors)) or 'none')
    return psas, neighbors


# ---- NBM data --------------------------------------------------------------------
def nbm_url(base, cycle, fhr):
    return (f'{base}/blend.{cycle:%Y%m%d}/{cycle:%H}/core/'
            f'blend.t{cycle:%H}z.core.f{fhr:03d}.co.grib2')


def get_idx(cycle, fhr):
    """Returns (grib_url, idx_text) from the first source that has the file, else (None, None)."""
    for base in NBM_SOURCES:
        url = nbm_url(base, cycle, fhr)
        try:
            r = requests.get(url + '.idx', timeout=30)
            if r.status_code == 200 and r.text.strip():
                return url, r.text
        except requests.RequestException:
            pass
    return None, None


def vrate_byte_range(idx_text):
    """Find the VRATE message in a GRIB2 .idx file -> (start, end) bytes; end may be None."""
    lines = [l for l in idx_text.strip().splitlines() if l.strip()]
    for i, line in enumerate(lines):
        parts = line.split(':')
        if len(parts) > 4 and parts[3] == 'VRATE' and 'std dev' not in line and 'prob' not in line:
            start = int(parts[1])
            end = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else None
            return start, end
    return None


def valid_times_for_day(day):
    base = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    return [base + dt.timedelta(hours=h) for h in AFTERNOON_UTC_HOURS]


def fhr_for(cycle, valid):
    return int((valid - cycle).total_seconds() // 3600)


def pick_cycle(days):
    """Newest NBM cycle that already has the Day 7 afternoon file posted."""
    now = dt.datetime.now(dt.timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = sorted(
        [today + dt.timedelta(hours=h) for h in (18, 12, 6, 0)] +
        [today - dt.timedelta(days=1) + dt.timedelta(hours=h) for h in (18, 12)],
        reverse=True)
    candidates = [c for c in candidates if c <= now - dt.timedelta(minutes=45)]

    last_valid = valid_times_for_day(days[-1])[-1]
    first_valid = valid_times_for_day(days[0])[0]
    for c in candidates:
        if get_idx(c, fhr_for(c, last_valid))[0]:
            return c
    for c in candidates:                       # fallback: at least Day 1 is posted
        f = fhr_for(c, first_valid)
        if f >= 1 and get_idx(c, f)[0]:
            return c
    raise RuntimeError('No recent NBM cycle found on AWS or NOMADS.')


class Grid:
    lats = lons = sl = None


def read_vrate(cycle, fhr):
    """Download just the VRATE message; returns knot-ft on the AZ/NM subset, or None."""
    if fhr < 1 or fhr > 264:
        return None
    url, idx = get_idx(cycle, fhr)
    if not url:
        print(f'  f{fhr:03d}: not posted')
        return None
    rng = vrate_byte_range(idx)
    if not rng:
        print(f'  f{fhr:03d}: no VRATE in this file')
        return None
    start, end = rng
    headers = {'Range': f'bytes={start}-{end if end is not None else ""}'}
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name
    try:
        grbs = pygrib.open(path)
        g = grbs.message(1)
        vals = np.ma.filled(np.ma.asarray(g.values, dtype=float), np.nan)
        if Grid.sl is None:
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
    vals[vals > 1e6] = np.nan                  # missing-value codes
    return vals * MPS_TO_KTFT


def daily_max(cycle, day):
    fields = []
    for valid in valid_times_for_day(day):
        f = fhr_for(cycle, valid)
        v = read_vrate(cycle, f)
        if v is not None:
            fields.append(v)
    if not fields:
        return None
    with np.errstate(all='ignore'):
        return np.fmax.reduce(fields)


# ---- Drawing ---------------------------------------------------------------------
def category(v):
    if v is None or np.isnan(v):
        return '—'
    for i in range(len(CAT_NAMES)):
        if v < CAT_BOUNDS[i + 1]:
            return CAT_NAMES[i]
    return CAT_NAMES[-1]


def day_label(day):
    return f'{day:%a} {day.month}/{day.day}'


def legend_panel(ax, cycle):
    ax.axis('off')
    handles = [Patch(facecolor=c, edgecolor='black',
                     label=f'{n}  ({CAT_BOUNDS[i] // 1000}K–{CAT_BOUNDS[i + 1] // 1000}K)'
                     if i < len(CAT_NAMES) - 1 else f'{n}  (≥{CAT_BOUNDS[i] // 1000}K)')
               for i, (n, c) in enumerate(zip(CAT_NAMES, CAT_COLORS))]
    ax.legend(handles=handles, loc='upper center', title='Ventilation (knot-ft)',
              fontsize=10, title_fontsize=11, frameon=False)
    ax.text(0.5, 0.18,
            f'NBM {cycle:%H}Z {cycle.month}/{cycle.day}/{cycle.year} run\n'
            f'Daily max of {AFTERNOON_TEXT}\n'
            f'Categories: NM smoke management',
            ha='center', va='center', fontsize=9, transform=ax.transAxes)


def draw_field(ax, data):
    if data is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, transform=ax.transAxes)
        return
    ax.pcolormesh(Grid.lons, Grid.lats, data, transform=ccrs.PlateCarree(),
                  cmap=cmap, norm=norm, shading='nearest', zorder=1)


def psa_mask(geom):
    return shapely.contains_xy(geom, Grid.lons, Grid.lats)


ROADS = None


def load_roads():
    """Interstates and US highways near AZ/NM from Natural Earth -> {(level, number): geometry}."""
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
            raw = str(a.get('level') or a.get('level_') or '').strip()
            if a.get('sov_a3') not in (None, 'USA'):
                continue
            seen_levels.add(raw)
            low = raw.lower()
            if 'interstate' in low:
                level = 'Interstate'
            elif 'federal' in low or low in ('us', 'u.s.'):
                level = 'Federal'
            else:
                continue
            geom = rec.geometry
            if geom is None or not geom.intersects(area):
                continue
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
    if geom.is_empty:
        return None
    if geom.geom_type == 'MultiLineString':
        geom = linemerge(geom)
    lines = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
    lines = [g for g in lines if g.geom_type == 'LineString']
    return max(lines, key=lambda g: g.length) if lines else None


def draw_roads(ax, extent, levels, fontsize):
    roads = load_roads()
    if not roads:
        return
    x0, x1, y0, y1 = extent
    view = shapely.box(x0, y0, x1, y1)
    style = {'Interstate': dict(color='#222222', linewidth=1.1),
             'Federal': dict(color='#444444', linewidth=0.6)}
    shield = {'Interstate': dict(facecolor='#1f4e9c', edgecolor='white', boxstyle='round,pad=0.25'),
              'Federal': dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2')}
    min_len = 0.08 * max(x1 - x0, y1 - y0)     # skip labels on tiny road pieces

    for level in reversed(levels):              # draw US highways first, interstates on top
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
                pt = line.interpolate(0.5, normalized=True)
                ax.text(pt.x, pt.y, num, transform=ccrs.PlateCarree(), fontsize=fontsize,
                        fontweight='bold', ha='center', va='center', zorder=6, clip_on=True,
                        color='white' if level == 'Interstate' else 'black', bbox=shield[level])


def draw_neighbors(ax, neighbors, extent, fontsize):
    """Bordering PSAs from other GACCs: gray dashed outline + gray code label (context only)."""
    if not neighbors:
        return
    ax.add_geometries([p['geom'] for p in neighbors], ccrs.PlateCarree(), facecolor='none',
                      edgecolor='#555555', linewidth=0.7, linestyle='--', zorder=3)
    x0, x1, y0, y1 = extent
    for p in neighbors:
        visible = p['geom'].intersection(shapely.box(x0, y0, x1, y1))
        if visible.is_empty or visible.area < 0.02:
            continue
        pt = visible.representative_point()
        ax.text(pt.x, pt.y, p['code'], transform=ccrs.PlateCarree(), fontsize=fontsize,
                color='#444444', fontstyle='italic', ha='center', va='center', zorder=5, clip_on=True,
                bbox=dict(facecolor='white', alpha=0.55, edgecolor='none', pad=0.4))


def make_overview(psas, neighbors, days, fields, cycle):
    fig = plt.figure(figsize=(16, 9))
    extent = OVERVIEW_EXTENT
    for i, (day, data) in enumerate(zip(days, fields)):
        ax = fig.add_subplot(2, 4, i + 1, projection=ccrs.Mercator())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        draw_field(ax, data)
        ax.add_feature(USSTATES.with_scale('5m'), edgecolor='black', linewidth=1.0, zorder=3)
        draw_neighbors(ax, neighbors, extent, fontsize=5)
        draw_roads(ax, extent, OVERVIEW_ROAD_LEVELS, fontsize=4.5)
        ax.add_geometries([p['geom'] for p in psas], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', linewidth=0.6, zorder=4)
        for p in psas:
            pt = p['geom'].representative_point()
            ax.text(pt.x, pt.y, p['code'], transform=ccrs.PlateCarree(), fontsize=5.5,
                    fontweight='bold', ha='center', va='center', zorder=5,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
        ax.set_title(f'Day {i + 1}: {day_label(day)}', fontsize=11, fontweight='bold')
    legend_panel(fig.add_subplot(2, 4, 8), cycle)
    fig.suptitle('NBM Max Afternoon Ventilation Rate — Arizona & New Mexico PSAs',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, 'vent_overview.png')
    fig.savefig(out, dpi=120, facecolor='white')
    plt.close(fig)
    print('saved', out)


def make_psa_map(psa, psas, neighbors, days, fields, cycle):
    minx, miny, maxx, maxy = psa['geom'].bounds
    pad = 0.15 * max(maxx - minx, maxy - miny) + 0.1
    extent = [minx - pad, maxx + pad, miny - pad, maxy + pad]
    mask = psa_mask(psa['geom'])
    others = [p['geom'] for p in psas if p is not psa]

    fig = plt.figure(figsize=(16, 9))
    for i, (day, data) in enumerate(zip(days, fields)):
        ax = fig.add_subplot(2, 4, i + 1, projection=ccrs.Mercator())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        draw_field(ax, data)
        ax.add_feature(USCOUNTIES.with_scale('500k'), edgecolor='gray', linewidth=0.4, zorder=2)
        ax.add_feature(USSTATES.with_scale('500k'), edgecolor='black', linewidth=1.0, zorder=3)
        draw_neighbors(ax, neighbors, extent, fontsize=7)
        draw_roads(ax, extent, PSA_ROAD_LEVELS, fontsize=6.5)
        if others:
            ax.add_geometries(others, ccrs.PlateCarree(), facecolor='none',
                              edgecolor='dimgray', linewidth=0.8, linestyle='--', zorder=3)
        ax.add_geometries([psa['geom']], ccrs.PlateCarree(), facecolor='none',
                          edgecolor='black', linewidth=2.2, zorder=4)

        if data is not None and mask.any():
            with np.errstate(all='ignore'):
                med = float(np.nanmedian(data[mask]))
            sub = f'Median {med:,.0f} kt-ft · {category(med)}' if not np.isnan(med) else 'No data'
        else:
            sub = 'No data'
        ax.set_title(f'Day {i + 1}: {day_label(day)}\n{sub}', fontsize=10, fontweight='bold')

    legend_panel(fig.add_subplot(2, 4, 8), cycle)
    title = f"{psa['code']} {psa['name']}".strip()
    fig.suptitle(f'{title} — NBM Max Afternoon Ventilation Rate', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, psa['file'])
    fig.savefig(out, dpi=120, facecolor='white')
    plt.close(fig)
    print('saved', out)


# ---- Webpage ---------------------------------------------------------------------
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
    stamp = f'{cycle:%Y%m%d%H}'                    # makes browsers load today's images, not cached ones
    built = dt.datetime.now(LOCAL_TZ)
    esc = html.escape

    nav = ['<a href="#overview">Overview</a>'] + [
        f'<a href="#{esc(re.sub(r"[^A-Za-z0-9]", "", p["code"]))}" title="{esc(p["name"])}">{esc(p["code"])}</a>'
        for p in psas]

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
    NBM {cycle:%H}Z {cycle.month}/{cycle.day}/{cycle.year} run ·
    daily max of {esc(AFTERNOON_TEXT)} · updated {built:%-I:%M %p} MST {built.month}/{built.day}</div>
</header>
<nav><div class="inner">{''.join(nav)}</div></nav>
<main>
{chr(10).join(sections)}
<p class="note">Ventilation rate = mixing height × transport wind, from the National Blend of Models (NBM),
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    psas, neighbors = load_psas()
    if not psas:
        sys.exit('No AZ/NM PSAs found.')

    today = dt.datetime.now(LOCAL_TZ).date()
    days = [today + dt.timedelta(days=i) for i in range(DAYS)]
    cycle = pick_cycle(days)
    print(f'Using NBM {cycle:%Y-%m-%d %H}Z')

    fields = []
    for day in days:
        print(f'Day {day}:')
        fields.append(daily_max(cycle, day))
    if all(f is None for f in fields):
        sys.exit('No NBM ventilation data could be read.')

    make_overview(psas, neighbors, days, fields, cycle)
    for psa in psas:
        make_psa_map(psa, psas, neighbors, days, fields, cycle)

    write_index_html(psas, days, cycle)


if __name__ == '__main__':
    main()
