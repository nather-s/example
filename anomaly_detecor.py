import pandas as pd
import numpy as np
import ast

POSITION_TYPES = ['PositionReport', 'StandardClassBPositionReport', 'ExtendedClassBPositionReport']
GAP_THRESHOLD_SEC = 720
TURN_THRESHOLD_DEG = 150
STOP_THRESHOLD_SEC = 1600
JUMP_THRESHOLD_KTS = 50
MIN_SIGNALS_TO_FLAG = 2
MIN_PINGS_FOR_CONFIDENCE = 3
MAX_BRIDGE_GAP_SEC = 300
PORT_RADIUS_NM = 2.0
PORT_DENSITY_THRESHOLD = 10

_SLOW_LAT = None
_SLOW_LON = None
_SLOW_MMSI = None


def load_data(filepath):
    df = pd.read_csv(filepath)
    positions = df[df['Message Type'].isin(POSITION_TYPES)].copy()
    positions['time'] = pd.to_datetime(positions['Received UTC'], format='%H:%M:%S')
    return positions.sort_values(['MMSI', 'time']).reset_index(drop=True)


def build_tracks(positions):
    return positions.groupby('MMSI')


def _circular_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R_nm * np.arcsin(np.sqrt(a))


def _build_port_lookup(positions):
    global _SLOW_LAT, _SLOW_LON, _SLOW_MMSI
    slow = positions[positions['Speed Over Ground (knots)'] < 2][['MMSI', 'Latitude', 'Longitude']].drop_duplicates('MMSI')
    _SLOW_LAT = slow['Latitude'].values
    _SLOW_LON = slow['Longitude'].values
    _SLOW_MMSI = slow['MMSI'].values


def _is_near_port(lat, lon, exclude_mmsi):
    if _SLOW_LAT is None:
        raise RuntimeError("_build_port_lookup(positions) must be called before scoring any tracks.")
    dists = _haversine_nm(lat, lon, _SLOW_LAT, _SLOW_LON)
    return ((dists <= PORT_RADIUS_NM) & (_SLOW_MMSI != exclude_mmsi)).sum() > PORT_DENSITY_THRESHOLD


def _gap_signal(track_df):
    t = track_df.sort_values('time')['time']
    if len(t) < 2:
        return 0.0
    return t.diff().dropna().max().total_seconds()


def _turn_signal(track_df):
    g = track_df.sort_values('time')
    moving = g[g['Speed Over Ground (knots)'] >= 1]
    course = moving['Course Over Ground'].dropna().values
    if len(course) < 2:
        return 0.0
    return max(_circular_diff(course[i], course[i - 1]) for i in range(1, len(course)))


def _stop_signal(track_df):
    g = track_df.sort_values('time')
    times = g['time'].tolist()
    speeds = g['Speed Over Ground (knots)'].tolist()
    max_dur, start = 0.0, None
    for i in range(len(times)):
        if speeds[i] < 1:
            if start is None:
                start = times[i]
            elif (times[i] - times[i - 1]).total_seconds() > MAX_BRIDGE_GAP_SEC:
                start = times[i]
            max_dur = max(max_dur, (times[i] - start).total_seconds())
        else:
            start = None
    return max_dur


def _position_jump_signal(track_df):
    g = track_df.sort_values('time')
    lat, lon, t = g['Latitude'].values, g['Longitude'].values, g['time'].values
    if len(g) < 2:
        return 0.0
    best = 0.0
    for i in range(1, len(g)):
        if lat[i] == 0 or lon[i] == 0 or lat[i - 1] == 0 or lon[i - 1] == 0:
            return 999.0
        dt_hr = (t[i] - t[i - 1]) / np.timedelta64(1, 'h')
        if dt_hr <= 0:
            continue
        best = max(best, _haversine_nm(lat[i - 1], lon[i - 1], lat[i], lon[i]) / dt_hr)
    return best


def score_track(track_df):
    n = len(track_df)
    if n < MIN_PINGS_FOR_CONFIDENCE:
        return {'low_confidence': True, 'n_pings': n, 'score': None,
                'is_anomalous': None, 'flags': ''}

    gap = _gap_signal(track_df)
    turn = _turn_signal(track_df)
    stop = _stop_signal(track_df)
    jump = _position_jump_signal(track_df)
    near_port = _is_near_port(track_df['Latitude'].iloc[0], track_df['Longitude'].iloc[0], track_df['MMSI'].iloc[0])

    flags = []
    if gap > GAP_THRESHOLD_SEC:
        flags.append(f'signal gap of {gap:.0f}s')
    if turn > TURN_THRESHOLD_DEG and not near_port:
        flags.append(f'sharp turn of {turn:.0f} deg (while moving, open water)')
    if stop > STOP_THRESHOLD_SEC:
        flags.append(f'long stop of {stop:.0f}s')
    if jump > JUMP_THRESHOLD_KTS:
        flags.append(f'implied speed of {jump:.0f} kts (physically implausible)')

    score = round(gap / GAP_THRESHOLD_SEC + (turn / TURN_THRESHOLD_DEG if not near_port else 0) +
                  stop / STOP_THRESHOLD_SEC + jump / JUMP_THRESHOLD_KTS, 2)

    return {'low_confidence': False, 'n_pings': n, 'score': score,
            'is_anomalous': len(flags) >= MIN_SIGNALS_TO_FLAG, 'flags': '; '.join(flags),
            'near_port': near_port}


def _parse_dimensions(dim_str):
    try:
        d = ast.literal_eval(dim_str)
        return d.get('A', 0) + d.get('B', 0), d.get('C', 0) + d.get('D', 0)
    except Exception:
        return None, None


def build_identity_lookup(raw_df):
    all_mmsi = raw_df['MMSI'].drop_duplicates()
    malformed_mmsi = set(all_mmsi[all_mmsi.astype(str).str.len() != 9])

    name_flicker_mmsi = set()
    ship_msgs = raw_df[raw_df['Message Type'] != 'StandardSearchAndRescueAircraftReport']
    for mmsi, g in ship_msgs.groupby('MMSI'):
        names = g['Ship Name'].dropna().unique()
        base_names = {n.split('[')[0].strip() for n in names}
        if len(base_names) > 1:
            name_flicker_mmsi.add(mmsi)

    static = raw_df[raw_df['Message Type'] == 'ShipStaticData'].dropna(subset=['Dimensions']).drop_duplicates('MMSI')
    lengths, beams, mmsis = [], [], []
    for _, row in static.iterrows():
        l, b = _parse_dimensions(row['Dimensions'])
        if l is None:
            continue
        lengths.append(l); beams.append(b); mmsis.append(row['MMSI'])
    dims = pd.DataFrame({'MMSI': mmsis, 'length': lengths, 'beam': beams})
    zero_dim_mmsi = set(dims[(dims['length'] == 0) & (dims['beam'] == 0)]['MMSI'])
    bad_hull_mmsi = set(dims[(dims['length'] > 0) & (dims['beam'] > dims['length'])]['MMSI'])

    rows = []
    for mmsi in all_mmsi:
        flags = []
        if mmsi in malformed_mmsi:
            flags.append('MMSI is not a valid 9-digit identifier')
        if mmsi in name_flicker_mmsi:
            flags.append('broadcast more than one distinct vessel name')
        if mmsi in zero_dim_mmsi:
            flags.append('vessel dimensions are missing or all zero')
        if mmsi in bad_hull_mmsi:
            flags.append('reported beam is wider than reported length (impossible hull)')
        rows.append({'MMSI': mmsi, 'identity_flagged': len(flags) > 0, 'identity_flags': '; '.join(flags)})

    return pd.DataFrame(rows).set_index('MMSI')


def score_all_tracks(tracks, positions, raw_df):
    _build_port_lookup(positions)
    identity_lookup = build_identity_lookup(raw_df)

    rows = []
    for mmsi, g in tracks:
        result = score_track(g)
        ship_name = g['Ship Name'].dropna().iloc[0] if g['Ship Name'].notna().any() else None

        if mmsi in identity_lookup.index:
            identity_flagged = bool(identity_lookup.loc[mmsi, 'identity_flagged'])
            identity_flags = identity_lookup.loc[mmsi, 'identity_flags']
        else:
            identity_flagged, identity_flags = False, ''

        rows.append({'MMSI': mmsi, 'Ship Name': ship_name,
                      'lat': g['Latitude'].iloc[0], 'lon': g['Longitude'].iloc[0],
                      **result, 'identity_flagged': identity_flagged, 'identity_flags': identity_flags})
    return pd.DataFrame(rows)


def save_results(results_df, filepath='full_anomaly_results.csv'):
    results_df.to_csv(filepath, index=False)
    return filepath
def get_time_bounds(filepath='formatted_ship_messages.csv'):
    """Returns the earliest and latest Received UTC timestamps in the file,
    as time strings, so the UI can build a slider/replay range from real data."""
    df = pd.read_csv(filepath)
    times = pd.to_datetime(df['Received UTC'], format='%H:%M:%S')
    return times.min().strftime('%H:%M:%S'), times.max().strftime('%H:%M:%S')


def score_as_of(cutoff_str, raw_df, full_positions):
    """Re-run the full detection pipeline using only data received up to
    `cutoff_str` (a 'HH:MM:SS' string). This simulates 'replaying' the window
    as if messages were arriving live, one at a time, up to that point."""
    cutoff = pd.to_datetime(cutoff_str, format='%H:%M:%S')
    raw_time = pd.to_datetime(raw_df['Received UTC'], format='%H:%M:%S')
    raw_subset = raw_df[raw_time <= cutoff]
    pos_subset = full_positions[full_positions['time'] <= cutoff]

    if pos_subset.empty:
        return pd.DataFrame(columns=['MMSI', 'Ship Name', 'lat', 'lon', 'low_confidence',
                                      'n_pings', 'score', 'is_anomalous', 'flags',
                                      'near_port', 'identity_flagged', 'identity_flags'])

    tracks_subset = build_tracks(pos_subset)
    return score_all_tracks(tracks_subset, pos_subset, raw_subset)

if __name__ == '__main__':
    raw_df = pd.read_csv('formatted_ship_messages.csv')
    positions = load_data('formatted_ship_messages.csv')
    tracks = build_tracks(positions)
    results = score_all_tracks(tracks, positions, raw_df)
    save_results(results)

    print(f"Total: {len(results):,}")
    print(f"Low confidence (movement): {results['low_confidence'].sum():,}")
    print(f"Anomalous (movement):      {(results['is_anomalous']==True).sum():,}")
    print(f"Identity/registration flagged: {results['identity_flagged'].sum():,}")
    print(f"  -- of which previously had NO movement verdict at all: "
          f"{(results['identity_flagged'] & results['low_confidence']).sum():,}")