"""
NEXUS Maritime Anomaly Detector
Multi-algorithm AIS anomaly detection with interactive dark dashboard
Run: streamlit run nexus_anomaly_detector.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# CONSTANTS & CONFIGURATION
# ================================================================

SHIP_TYPE_SPEED_LIMITS = {
    (20, 29): (0, 15, "WIG/Special"),
    (30, 39): (0, 12, "Fishing"),
    (40, 49): (0, 15, "Towing"),
    (50, 59): (0, 15, "Towing Large"),
    (60, 69): (0, 28, "Passenger"),
    (70, 79): (0, 22, "Cargo"),
    (80, 89): (0, 18, "Tanker"),
    (90, 99): (0, 15, "Other"),
}

SHIP_TYPE_RANGES = [
    (range(20, 30), "WIG/Special"),
    (range(30, 40), "Fishing"),
    (range(40, 50), "Towing"),
    (range(50, 60), "Towing Large"),
    (range(60, 70), "Passenger"),
    (range(70, 80), "Cargo"),
    (range(80, 90), "Tanker"),
    (range(90, 100), "Other"),
]

SEVERITY_COLORS = {
    "CRITICAL": "#ff2d55",
    "HIGH": "#ff9500",
    "MEDIUM": "#ffcc00",
    "LOW": "#30d158",
}

ANOMALY_CONFIG = {
    "Speed Zone Violation":     {"color": "#ff2d55", "icon": "⚡", "sev": "HIGH"},
    "Heading-Course Divergence":{"color": "#ff9500", "icon": "🧭", "sev": "MEDIUM"},
    "Ghost/Anonymous Vessel":  {"color": "#af52de", "icon": "👻", "sev": "HIGH"},
    "Proximity Clustering":    {"color": "#ff3b30", "icon": "⚠️", "sev": "CRITICAL"},
    "Erratic Course Change":   {"color": "#ffcc00", "icon": "🔄", "sev": "MEDIUM"},
    "Speed Inconsistency":     {"color": "#5ac8fa", "icon": "📊", "sev": "HIGH"},
    "Small Vessel High Speed": {"color": "#ff6b6b", "icon": "🚤", "sev": "CRITICAL"},
}

HEADING_NA = 511.0

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def get_ship_type_label(ship_type):
    if pd.isna(ship_type):
        return "Unknown"
    try:
        n = float(ship_type)
        for rng, label in SHIP_TYPE_RANGES:
            if n in rng:
                return label
    except (ValueError, TypeError):
        pass
    return "Unknown"


def get_speed_limit(ship_type):
    if pd.isna(ship_type):
        return 30.0
    try:
        n = float(ship_type)
        for (lo, hi), (_, mx, _) in SHIP_TYPE_SPEED_LIMITS.items():
            if lo <= n <= hi:
                return mx
    except (ValueError, TypeError):
        pass
    return 30.0


def haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(min(1.0, a)))


def angle_diff(a, b):
    d = (a - b) % 360
    return d if d <= 180 else 360 - d


def safe_name(val):
    if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
        return ""
    return str(val).strip()


# ================================================================
# DATA LOADING & PREPROCESSING
# ================================================================

@st.cache_data(ttl=300)
def load_data(filepath="formatted_ship_messages.csv"):
    raw = pd.read_csv(filepath)

    pos_mask = raw["Message Type"].isin(["PositionReport", "StandardClassBPositionReport"]) & (raw["Is Valid"] == True)
    pos = raw[pos_mask].copy()

    for c in ["Latitude", "Longitude", "Speed Over Ground (knots)",
              "Course Over Ground", "True Heading", "Timestamp (sec)"]:
        pos[c] = pd.to_numeric(pos[c], errors="coerce")

    pos = pos[
        pos["Latitude"].between(-90, 90) &
        pos["Longitude"].between(-180, 180) &
        pos["Speed Over Ground (knots)"].between(0, 60)
    ].copy()

    static = raw[raw["Message Type"].isin(["ShipStaticData", "StaticDataReport"])].copy()
    info = {}
    for _, r in static.iterrows():
        m = r["MMSI"]
        n = safe_name(r.get("Ship Name"))
        t = r.get("Ship Type")
        cs = safe_name(r.get("Call Sign"))
        dst = safe_name(r.get("Destination"))
        dim_str = str(r.get("Dimensions", ""))

        if n:
            info.setdefault(m, {})["name"] = n
        if pd.notna(t):
            try:
                info.setdefault(m, {})["type"] = float(t)
            except (ValueError, TypeError):
                pass
        if cs:
            info.setdefault(m, {})["callsign"] = cs
        if dst:
            info.setdefault(m, {})["destination"] = dst
        if dim_str not in ("", "nan", "None"):
            try:
                d = json.loads(dim_str.replace("'", '"'))
                info.setdefault(m, {})["len_a"] = d.get("A", 0)
                info.setdefault(m, {})["len_b"] = d.get("B", 0)
                info.setdefault(m, {})["wid_c"] = d.get("C", 0)
                info.setdefault(m, {})["wid_d"] = d.get("D", 0)
            except (json.JSONDecodeError, TypeError):
                pass

    for col in ["name", "callsign", "destination"]:
        pos[col] = None
    for col in ["type", "len_a", "len_b", "wid_c", "wid_d"]:
        pos[col] = np.nan

    for mmsi, d in info.items():
        mask = pos["MMSI"] == mmsi
        for k, v in d.items():
            pos.loc[mask, k] = v

    pos["total_length"] = pos["len_a"].fillna(0) + pos["len_b"].fillna(0)
    pos = pos.sort_values(["MMSI", "Timestamp (sec)"]).reset_index(drop=True)
    return pos


# ================================================================
# DETECTION ALGORITHMS  (7 total)
# ================================================================

def alg_speed_zone(df):
    out = []
    for _, r in df.iterrows():
        spd = r["Speed Over Ground (knots)"]
        lim = get_speed_limit(r.get("type"))
        lbl = get_ship_type_label(r.get("type"))
        if spd > lim:
            exc = spd - lim
            sev = "CRITICAL" if exc > 10 else ("HIGH" if exc > 5 else "MEDIUM")
            out.append(dict(
                mmsi=r["MMSI"], ship_name=safe_name(r.get("name")),
                anomaly_type="Speed Zone Violation", severity=sev,
                latitude=r["Latitude"], longitude=r["Longitude"],
                speed=spd, timestamp=str(r["Received UTC"]),
                detail=f"Speed {spd:.1f}kn exceeds {lbl} limit ({lim:.0f}kn) by {exc:.1f}kn",
                score=min(exc / 10.0, 1.0),
            ))
    return out


def alg_heading_course_div(df):
    out = []
    for _, r in df.iterrows():
        h = r.get("True Heading", HEADING_NA)
        c = r.get("Course Over Ground", 0)
        s = r["Speed Over Ground (knots)"]
        if pd.isna(h) or h == HEADING_NA or s < 1.0:
            continue
        diff = angle_diff(h, c)
        if diff > 30:
            sev = "CRITICAL" if diff > 60 else ("HIGH" if diff > 45 else "MEDIUM")
            out.append(dict(
                mmsi=r["MMSI"], ship_name=safe_name(r.get("name")),
                anomaly_type="Heading-Course Divergence", severity=sev,
                latitude=r["Latitude"], longitude=r["Longitude"],
                speed=s, timestamp=str(r["Received UTC"]),
                detail=f"Heading-Course divergence {diff:.1f}° (Hdg {h:.0f}° / CoG {c:.0f}°) at {s:.1f}kn — possible drift or towing",
                score=min(diff / 90.0, 1.0),
            ))
    return out


def alg_ghost_vessel(df):
    out = []
    for _, r in df.iterrows():
        flags = sum([
            not safe_name(r.get("name")),
            pd.isna(r.get("type")),
            pd.isna(r.get("True Heading")) or r.get("True Heading") == HEADING_NA,
            not safe_name(r.get("callsign")),
        ])
        if flags >= 3:
            sev = "CRITICAL" if flags >= 4 else "HIGH"
            missing = []
            if not safe_name(r.get("name")):   missing.append("name")
            if pd.isna(r.get("type")):         missing.append("type")
            if pd.isna(r.get("True Heading")) or r.get("True Heading") == HEADING_NA: missing.append("heading")
            if not safe_name(r.get("callsign")): missing.append("callsign")
            out.append(dict(
                mmsi=r["MMSI"],
                ship_name=safe_name(r.get("name")) or "[ANONYMOUS]",
                anomaly_type="Ghost/Anonymous Vessel", severity=sev,
                latitude=r["Latitude"], longitude=r["Longitude"],
                speed=r["Speed Over Ground (knots)"],
                timestamp=str(r["Received UTC"]),
                detail=f"Missing identity: {', '.join(missing)} ({flags}/4) — potential spoofed/hidden vessel",
                score=flags / 4.0,
            ))
    return out


def alg_proximity(df, threshold_nm=0.3):
    out = []
    last = df.groupby("MMSI").last().reset_index()
    rows = last[["MMSI", "Latitude", "Longitude",
                  "Speed Over Ground (knots)", "Received UTC", "name"]].values
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_nm(rows[i][1], rows[i][2], rows[j][1], rows[j][2])
            if 0.001 < d < threshold_nm:
                sev = "CRITICAL" if d < 0.1 else ("HIGH" if d < 0.2 else "MEDIUM")
                out.append(dict(
                    mmsi=int(rows[i][0]), ship_name=safe_name(rows[i][5]),
                    anomaly_type="Proximity Clustering", severity=sev,
                    latitude=(rows[i][1] + rows[j][1]) / 2,
                    longitude=(rows[i][2] + rows[j][2]) / 2,
                    speed=(rows[i][3] + rows[j][3]) / 2,
                    timestamp=str(rows[i][4]),
                    detail=f"Vessels {int(rows[i][0])} & {int(rows[j][0])} only {d:.3f}nm apart — collision risk / suspicious meeting",
                    score=max(0, 1.0 - d / threshold_nm),
                ))
    return out


def alg_erratic_course(df):
    out = []
    col_idx = {c: df.columns.get_loc(c) for c in df.columns}
    for mmsi, grp in df.groupby("MMSI"):
        if len(grp) < 2:
            continue
        arr = grp.sort_values("Timestamp (sec)").values
        for i in range(1, len(arr)):
            pc = arr[i - 1][col_idx["Course Over Ground"]]
            cc = arr[i][col_idx["Course Over Ground"]]
            ps = arr[i - 1][col_idx["Speed Over Ground (knots)"]]
            cs = arr[i][col_idx["Speed Over Ground (knots)"]]
            if pd.isna(pc) or pd.isna(cc) or (ps < 2 and cs < 2):
                continue
            diff = angle_diff(cc, pc)
            if diff > 90:
                sev = "CRITICAL" if diff > 150 else ("HIGH" if diff > 120 else "MEDIUM")
                out.append(dict(
                    mmsi=int(mmsi),
                    ship_name=safe_name(arr[i][col_idx["name"]]),
                    anomaly_type="Erratic Course Change", severity=sev,
                    latitude=arr[i][col_idx["Latitude"]],
                    longitude=arr[i][col_idx["Longitude"]],
                    speed=cs, timestamp=str(arr[i][col_idx["Received UTC"]]),
                    detail=f"Course shift {diff:.1f}° ({pc:.0f}° → {cc:.0f}°) between reports — erratic maneuvering",
                    score=min(diff / 180.0, 1.0),
                ))
    return out


def alg_speed_inconsistency(df):
    out = []
    col_idx = {c: df.columns.get_loc(c) for c in df.columns}
    for mmsi, grp in df.groupby("MMSI"):
        if len(grp) < 2:
            continue
        arr = grp.sort_values("Timestamp (sec)").values
        for i in range(1, len(arr)):
            plat, plon = arr[i - 1][col_idx["Latitude"]], arr[i - 1][col_idx["Longitude"]]
            clat, clon = arr[i][col_idx["Latitude"]], arr[i][col_idx["Longitude"]]
            pts, cts = arr[i - 1][col_idx["Timestamp (sec)"]], arr[i][col_idx["Timestamp (sec)"]]
            rep = arr[i][col_idx["Speed Over Ground (knots)"]]
            if any(pd.isna(x) for x in [plat, clat, pts, cts]):
                continue
            dt = cts - pts
            if dt < 1:
                continue
            calc = (haversine_nm(plat, plon, clat, clon) / dt) * 3600
            if rep < 0.5:
                continue
            delta = abs(calc - rep)
            if delta > 5.0:
                sev = "CRITICAL" if delta > 15 else ("HIGH" if delta > 10 else "MEDIUM")
                out.append(dict(
                    mmsi=int(mmsi),
                    ship_name=safe_name(arr[i][col_idx["name"]]),
                    anomaly_type="Speed Inconsistency", severity=sev,
                    latitude=clat, longitude=clon, speed=rep,
                    timestamp=str(arr[i][col_idx["Received UTC"]]),
                    detail=f"Reported {rep:.1f}kn vs calculated {calc:.1f}kn (Δ {delta:.1f}kn) — possible spoofed data",
                    score=min(delta / 20.0, 1.0),
                ))
    return out


def alg_small_fast(df):
    out = []
    for _, r in df.iterrows():
        length = r.get("total_length", np.nan)
        spd = r["Speed Over Ground (knots)"]
        if pd.isna(length) or length <= 0:
            continue
        if length < 20 and spd > 15:
            lbl = get_ship_type_label(r.get("type"))
            sev = "CRITICAL" if spd > 25 else ("HIGH" if spd > 20 else "MEDIUM")
            out.append(dict(
                mmsi=r["MMSI"], ship_name=safe_name(r.get("name")),
                anomaly_type="Small Vessel High Speed", severity=sev,
                latitude=r["Latitude"], longitude=r["Longitude"],
                speed=spd, timestamp=str(r["Received UTC"]),
                detail=f"{length:.0f}m {lbl} at {spd:.1f}kn — potential fast boat / smuggling activity",
                score=min((spd - 15) / 15.0, 1.0),
            ))
    return out


# ================================================================
# ORCHESTRATOR
# ================================================================

@st.cache_data(ttl=300)
def run_detections(df, prox_nm=0.3):
    all_a = []
    summary = {}
    algos = [
        ("Speed Zone Violation",      lambda: alg_speed_zone(df)),
        ("Heading-Course Divergence", lambda: alg_heading_course_div(df)),
        ("Ghost/Anonymous Vessel",    lambda: alg_ghost_vessel(df)),
        ("Proximity Clustering",      lambda: alg_proximity(df, prox_nm)),
        ("Erratic Course Change",     lambda: alg_erratic_course(df)),
        ("Speed Inconsistency",       lambda: alg_speed_inconsistency(df)),
        ("Small Vessel High Speed",   lambda: alg_small_fast(df)),
    ]
    for name, fn in algos:
        res = fn()
        all_a.extend(res)
        summary[name] = {
            "count": len(res),
            "CRITICAL": sum(1 for a in res if a["severity"] == "CRITICAL"),
            "HIGH": sum(1 for a in res if a["severity"] == "HIGH"),
            "MEDIUM": sum(1 for a in res if a["severity"] == "MEDIUM"),
        }
    best = {}
    for a in all_a:
        k = (a["mmsi"], a["anomaly_type"])
        if k not in best or a["score"] > best[k]["score"]:
            best[k] = a
    return list(best.values()), summary


# ================================================================
# CHART BUILDERS
# ================================================================

def _build_track_arrays(df, mmsi_set):
    """Efficiently build lat/lon arrays with NaN breaks between vessels."""
    lats, lons = [], []
    sub = df[df["MMSI"].isin(mmsi_set)]
    for _, grp in sub.groupby("MMSI"):
        if len(grp) < 2:
            continue
        g = grp.sort_values("Timestamp (sec)")
        lats.extend(g["Latitude"].tolist() + [None])
        lons.extend(g["Longitude"].tolist() + [None])
    return lats, lons


def chart_map(pos_df, anom_df):
    fig = go.Figure()
    anom_mmsis = set(anom_df["mmsi"].unique()) if len(anom_df) else set()

    # --- Layer 1: Normal vessel tracks (faint lines, single trace with NaN breaks) ---
    normal_df = pos_df[~pos_df["MMSI"].isin(anom_mmsis)]
    n_lats, n_lons = _build_track_arrays(normal_df, set(normal_df["MMSI"].unique()))
    if n_lats:
        fig.add_trace(go.Scattermapbox(
            lat=n_lats, lon=n_lons, mode="lines",
            line=dict(width=1, color="rgba(58,69,86,0.35)"),
            hoverinfo="none", name="Normal Tracks", showlegend=True,
        ))

    # --- Layer 2: Normal vessel last positions (dots) ---
    last_normal = normal_df.groupby("MMSI").last().reset_index()
    if len(last_normal):
        hover_texts = []
        for _, r in last_normal.iterrows():
            nm = r["name"] if r["name"] and str(r["name"]) != "nan" else ""
            hover_texts.append(f"MMSI: {r['MMSI']}{('<br>' + nm) if nm else ''}")
        fig.add_trace(go.Scattermapbox(
            lat=last_normal["Latitude"], lon=last_normal["Longitude"], mode="markers",
            marker=dict(size=3.5, color="#3a4556", opacity=0.55),
            text=hover_texts, hoverinfo="text", name="Normal Vessels", showlegend=True,
        ))

    # --- Layer 3: Anomalous vessel tracks (colored by severity) ---
    for sev in ("CRITICAL", "HIGH", "MEDIUM"):
        sev_mmsis = set(anom_df[anom_df["severity"] == sev]["mmsi"].unique())
        if not sev_mmsis:
            continue
        a_lats, a_lons = _build_track_arrays(pos_df, sev_mmsis)
        if a_lats:
            fig.add_trace(go.Scattermapbox(
                lat=a_lats, lon=a_lons, mode="lines",
                line=dict(width=3, color=SEVERITY_COLORS[sev]),
                hoverinfo="none", name=f"⚠ {sev} Track", showlegend=True,
            ))

    # --- Layer 4: Anomalous vessel markers (no `line` — not supported in scattermapbox) ---
    for sev in ("CRITICAL", "HIGH", "MEDIUM"):
        sub = anom_df[anom_df["severity"] == sev]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=sub["latitude"], lon=sub["longitude"], mode="markers",
            marker=dict(
                size=14 if sev == "CRITICAL" else (10 if sev == "HIGH" else 7),
                color=SEVERITY_COLORS[sev], opacity=0.95,
            ),
            text=sub["ship_name"] + "<br>" + sub["anomaly_type"] + "<br>" + sub["detail"],
            hoverinfo="text", name=f"⚠ {sev}", showlegend=True,
        ))

    clat = pos_df["Latitude"].mean()
    clon = pos_df["Longitude"].mean()
    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=clat, lon=clon), zoom=5),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(0,0,0,0.8)", font=dict(color="white", size=10),
            orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
        ),
    )
    return fig


def chart_radar(summary, active):
    cats = [k for k in summary if k in active]
    if not cats:
        return go.Figure()
    vals = [summary[c]["count"] for c in cats]
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=cats_c, fill="toself",
        fillcolor="rgba(255,45,85,0.12)", line=dict(color="#ff2d55", width=2),
        marker=dict(size=5, color="#ff2d55"),
        text=[f"{c}: {v}" for c, v in zip(cats, vals)], hoverinfo="text",
    ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                                   tickfont=dict(color="#666", size=8)),
                   angularaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                                    tickfont=dict(color="#aaa", size=7.5),
                                    rotation=30, direction="clockwise")),
        margin=dict(l=55, r=55, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Detection Radar", font=dict(color="#bbb", size=12), x=0.5),
    )
    return fig


def chart_severity_bars(adf):
    order = ["CRITICAL", "HIGH", "MEDIUM"]
    cnts = [len(adf[adf["severity"] == s]) for s in order]
    fig = go.Figure(data=[go.Bar(
        x=order, y=cnts,
        marker_color=[SEVERITY_COLORS[s] for s in order],
        marker_line_color="rgba(255,255,255,0.25)", marker_line_width=1,
        text=cnts, textposition="auto",
        textfont=dict(color="white", size=13, family="JetBrains Mono"),
    )])
    fig.update_layout(
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#777", size=9)),
        xaxis=dict(tickfont=dict(color="#bbb", size=10)),
        margin=dict(l=35, r=15, t=5, b=25), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", bargap=0.45,
    )
    return fig


def chart_speed_hist(pos_df, adf):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pos_df["Speed Over Ground (knots)"], nbinsx=30,
                               marker_color="rgba(58,69,86,0.55)", marker_line_color="rgba(58,69,86,0.8)",
                               name="All Vessels", opacity=0.6))
    if len(adf):
        am = set(adf["mmsi"].unique())
        fig.add_trace(go.Histogram(
            x=pos_df[pos_df["MMSI"].isin(am)]["Speed Over Ground (knots)"],
            nbinsx=30, marker_color="rgba(255,45,85,0.55)",
            marker_line_color="rgba(255,45,85,0.85)", name="Anomalous", opacity=0.7))
    fig.update_layout(barmode="overlay",
                      xaxis=dict(title=dict(text="Speed (kn)", font=dict(color="#777", size=10)),
                                 tickfont=dict(color="#777", size=9), gridcolor="rgba(255,255,255,0.04)"),
                      yaxis=dict(title=dict(text="Count", font=dict(color="#777", size=10)),
                                 tickfont=dict(color="#777", size=9), gridcolor="rgba(255,255,255,0.04)"),
                      margin=dict(l=35, r=15, t=5, b=35), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white", size=9),
                                  orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def chart_type_pie(adf, pos_df):
    if len(adf) == 0:
        return go.Figure()
    am = set(adf["mmsi"].unique())
    sub = pos_df[pos_df["MMSI"].isin(am)].drop_duplicates("MMSI").copy()
    sub["tl"] = sub["type"].apply(get_ship_type_label)
    vc = sub["tl"].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=vc.index, values=vc.values, hole=0.58,
        marker_colors=["#ff2d55", "#ff9500", "#ffcc00", "#5ac8fa", "#af52de", "#30d158", "#ff6b6b", "#8e8e93"],
        textfont=dict(color="white", size=10), textinfo="label+percent",
    )])
    fig.update_layout(margin=dict(l=15, r=15, t=5, b=15),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(color="white", size=9),
                                  orientation="h", yanchor="bottom", y=-0.08))
    return fig


def chart_algo_hbar(summary):
    names = list(summary.keys())
    cnts = [summary[n]["count"] for n in names]
    cols = [ANOMALY_CONFIG.get(n, {}).get("color", "#888") for n in names]
    fig = go.Figure(data=[go.Bar(
        y=names, x=cnts, orientation="h", marker_color=cols,
        marker_line_color="rgba(255,255,255,0.15)", marker_line_width=1,
        text=cnts, textposition="auto",
        textfont=dict(color="white", size=10, family="JetBrains Mono"),
    )])
    fig.update_layout(
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#777", size=9)),
        yaxis=dict(tickfont=dict(color="#bbb", size=9.5), autorange="reversed"),
        margin=dict(l=8, r=35, t=5, b=15), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=280,
    )
    return fig


# ================================================================
# CUSTOM CSS
# ================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

:root{
  --bg0:#0a0e17;--bg1:#111827;--bg2:#1a2235;--bdr:#1e2d45;
  --t1:#e2e8f0;--t2:#8892a4;--red:#ff2d55;--org:#ff9500;--ylw:#ffcc00;--grn:#30d158;--blu:#5ac8fa;--pur:#af52de;
}
.stApp{background:var(--bg0)!important;color:var(--t1)!important}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden!important}
.block-container{padding-top:.8rem!important;padding-bottom:2rem!important;max-width:1620px!important}

.mc{background:var(--bg2);border:1px solid var(--bdr);border-radius:12px;padding:1.1rem 1.4rem;text-align:center;position:relative;overflow:hidden;transition:all .25s}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.mc.r::before{background:var(--red)}.mc.o::before{background:var(--org)}
.mc.y::before{background:var(--ylw)}.mc.g::before{background:var(--grn)}.mc.b::before{background:var(--blu)}
.mc:hover{border-color:rgba(255,255,255,.13);transform:translateY(-2px)}
.mv{font-family:'JetBrains Mono',monospace;font-size:2.1rem;font-weight:600;line-height:1.1}
.ml{font-family:'Inter',sans-serif;font-size:.72rem;font-weight:600;color:var(--t2);text-transform:uppercase;letter-spacing:.09em;margin-top:.35rem}

.pnl{background:var(--bg2);border:1px solid var(--bdr);border-radius:12px;padding:1.1rem;margin-bottom:.9rem}
.sh{font-family:'Inter',sans-serif;font-size:.72rem;font-weight:600;color:var(--t2);text-transform:uppercase;letter-spacing:.11em;padding-bottom:.55rem;border-bottom:1px solid var(--bdr);margin-bottom:.9rem}

.at{width:100%;border-collapse:separate;border-spacing:0;font-family:'Inter',sans-serif;font-size:.82rem}
.at thead th{background:rgba(255,255,255,.025);color:var(--t2);font-weight:600;text-transform:uppercase;font-size:.68rem;letter-spacing:.07em;padding:.65rem .7rem;text-align:left;border-bottom:1px solid var(--bdr);position:sticky;top:0;z-index:2}
.at tbody tr{transition:background .15s}.at tbody tr:hover{background:rgba(255,255,255,.025)}
.at td{padding:.55rem .7rem;border-bottom:1px solid rgba(255,255,255,.025);color:var(--t1)}

.sb{display:inline-block;padding:.12rem .55rem;border-radius:20px;font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.sb-CRITICAL{background:rgba(255,45,85,.18);color:var(--red)}
.sb-HIGH{background:rgba(255,149,0,.18);color:var(--org)}
.sb-MEDIUM{background:rgba(255,204,0,.18);color:var(--ylw)}

.scbar{height:4px;border-radius:2px;background:rgba(255,255,255,.08);overflow:hidden;min-width:55px}
.scfill{height:100%;border-radius:2px;transition:width .3s}

[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--bdr)!important}
[data-testid="stSidebar"] .stMarkdown{color:var(--t1)!important}
.stSelectbox label,.stSlider label,.stMultiselect label{color:var(--t2)!important;font-family:'Inter',sans-serif!important;font-size:.78rem!important}
.stSelectbox>div>div{background:var(--bg2)!important;border-color:var(--bdr)!important;color:var(--t1)!important}
.stMultiselect>div>div{background:var(--bg2)!important;border-color:var(--bdr)!important}

.stTabs [data-baseweb="tab-list"]{gap:.2rem;background:rgba(0,0,0,.18);border-radius:8px;padding:.15rem}
.stTabs [data-baseweb="tab"]{border-radius:6px!important;font-family:'Inter',sans-serif!important;font-size:.78rem!important;color:var(--t2)!important;padding:.35rem .9rem!important}
.stTabs [aria-selected="true"]{background:var(--bg2)!important;color:var(--t1)!important;font-weight:600!important}
.stTabs [data-baseweb="tab-highlight"]{display:none!important}

.tb{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.3rem;padding-bottom:.9rem;border-bottom:1px solid var(--bdr)}
.tm{font-family:'Inter',sans-serif;font-weight:700;font-size:1.45rem;color:var(--t1);letter-spacing:-.02em}
.ts{font-family:'Inter',sans-serif;font-weight:400;font-size:.78rem;color:var(--t2)}
.ta{color:var(--red)}

.ld{width:8px;height:8px;background:var(--grn);border-radius:50%;display:inline-block;animation:pls 1.4s ease-in-out infinite}
@keyframes pls{0%,100%{opacity:1}50%{opacity:.5}}

.ar{display:flex;align-items:center;gap:.5rem;padding:.35rem 0;font-size:.8rem}
.ad{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.ad.on{background:var(--grn);box-shadow:0 0 6px var(--grn)}.ad.off{background:var(--t2);opacity:.35}

.tv{display:flex;align-items:center;gap:.75rem;padding:.45rem 0;border-bottom:1px solid rgba(255,255,255,.025)}

::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg0)}
::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.18)}

.stDownloadButton>button{background:var(--bg2)!important;border:1px solid var(--bdr)!important;color:var(--t1)!important;border-radius:8px!important;font-family:'Inter',sans-serif!important}
.stDownloadButton>button:hover{border-color:rgba(255,255,255,.2)!important;background:rgba(255,255,255,.04)!important}
</style>
"""


# ================================================================
# UI RENDERERS
# ================================================================

def metric_card(val, label, cls, vc):
    st.markdown(f'<div class="mc {cls}"><div class="mv" style="color:{vc}">{val}</div><div class="ml">{label}</div></div>',
                unsafe_allow_html=True)


def algo_status_panel(summary, active):
    st.markdown('<div class="sh">Detection Algorithms</div>', unsafe_allow_html=True)
    for name, info in summary.items():
        on = name in active
        st.markdown(f'''<div class="ar">
            <span class="ad {"on" if on else "off"}"></span>
            <span style="color:{"#e2e8f0" if on else "#444"}">{name}</span>
            <span style="color:{"#e2e8f0" if on else "#444"};font-family:JetBrains Mono,monospace;margin-left:auto">{info["count"]}</span>
        </div>''', unsafe_allow_html=True)


def anomaly_table(adf):
    if len(adf) == 0:
        st.markdown('<div style="text-align:center;color:#555;padding:2rem">No anomalies match current filters</div>', unsafe_allow_html=True)
        return
    rows = ""
    for _, r in adf.iterrows():
        sc = "#ff2d55" if r["score"] > .7 else ("#ff9500" if r["score"] > .4 else "#ffcc00")
        tc = ANOMALY_CONFIG.get(r["anomaly_type"], {}).get("color", "#ccc")
        ic = ANOMALY_CONFIG.get(r["anomaly_type"], {}).get("icon", "⚠️")
        nm = r["ship_name"] if r["ship_name"] else "—"
        rows += f'''<tr>
            <td style="font-family:JetBrains Mono,monospace;font-size:.76rem;color:#777">{r["mmsi"]}</td>
            <td style="font-weight:500">{nm}</td>
            <td><span class="sb sb-{r["severity"]}">{r["severity"]}</span></td>
            <td><span style="color:{tc}">{ic} {r["anomaly_type"]}</span></td>
            <td style="font-family:JetBrains Mono,monospace;font-size:.78rem">{r["speed"]:.1f}kn</td>
            <td style="font-size:.76rem;color:#888;max-width:280px">{r["detail"]}</td>
            <td><div class="scbar"><div class="scfill" style="width:{r["score"]*100:.0f}%;background:{sc}"></div></div></td>
            <td style="font-size:.72rem;color:#555">{r["timestamp"]}</td>
        </tr>'''
    st.markdown(f'''<div style="max-height:520px;overflow-y:auto;border-radius:8px;border:1px solid var(--bdr)">
        <table class="at"><thead><tr>
            <th>MMSI</th><th>Vessel</th><th>Severity</th><th>Type</th><th>Speed</th><th>Detail</th><th>Score</th><th>Time</th>
        </tr></thead><tbody>{rows}</tbody></table></div>''', unsafe_allow_html=True)


def top_vessels(adf):
    if len(adf) == 0:
        st.markdown('<div style="text-align:center;color:#555;padding:1.5rem">No anomalies</div>', unsafe_allow_html=True)
        return
    sev_rank = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}
    top = adf.groupby("mmsi").agg(
        name=("ship_name", "first"), cnt=("anomaly_type", "count"),
        ms=("score", "max"), types=("anomaly_type", lambda x: ", ".join(sorted(x.unique()))),
        msv=("severity", lambda x: max(x, key=lambda s: sev_rank.get(s, 0))),
    ).sort_values("ms", ascending=False).head(12)
    for mmsi, r in top.iterrows():
        sc = SEVERITY_COLORS.get(r["msv"], "#888")
        st.markdown(f'''<div class="tv">
            <span class="sb sb-{r["msv"]}" style="min-width:62px;text-align:center">{r["msv"]}</span>
            <div style="flex:1">
                <div style="font-weight:500;font-size:.83rem">{r["name"] if r["name"] else "—"}</div>
                <div style="font-size:.68rem;color:#555;font-family:JetBrains Mono,monospace">MMSI {mmsi}</div>
            </div>
            <div style="text-align:right">
                <div style="font-family:JetBrains Mono,monospace;font-size:1.05rem;font-weight:600;color:{sc}">{r["ms"]:.2f}</div>
                <div style="font-size:.67rem;color:#555">{r["cnt"]} flag{"s" if r["cnt"]>1 else ""}</div>
            </div>
        </div>''', unsafe_allow_html=True)


# ================================================================
# MAIN APP
# ================================================================

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown('''<div class="tb">
        <div><div class="tm">NEXUS <span class="ta">Anomaly Detector</span></div>
        <div class="ts">Mediterranean Maritime Traffic — Multi-Algorithm AIS Analysis</div></div>
        <div style="display:flex;align-items:center;gap:.55rem">
            <span class="ld"></span><span style="font-size:.75rem;color:var(--grn);font-weight:500">LIVE</span>
        </div></div>''', unsafe_allow_html=True)

    try:
        pos = load_data("formatted_ship_messages.csv")
    except FileNotFoundError:
        st.error("❌  formatted_ship_messages.csv not found — place it alongside this script.")
        return

    n_vessels = pos["MMSI"].nunique()
    n_msgs = len(pos)

    with st.sidebar:
        st.markdown('<div style="margin-bottom:1.2rem"><span style="font-family:Inter,sans-serif;font-weight:700;font-size:1.05rem;color:#e2e8f0">Controls</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="sh">Anomaly Types</div>', unsafe_allow_html=True)
        all_types = list(ANOMALY_CONFIG.keys())
        sel_types = st.multiselect("Types", all_types, default=all_types, label_visibility="collapsed")

        st.markdown('<div class="sh">Severity</div>', unsafe_allow_html=True)
        sel_sev = st.multiselect("Severity", ["CRITICAL", "HIGH", "MEDIUM"], default=["CRITICAL", "HIGH", "MEDIUM"], label_visibility="collapsed")

        st.markdown('<div class="sh">Ship Type</div>', unsafe_allow_html=True)
        present = sorted(set(get_ship_type_label(t) for t in pos["type"].dropna().unique()))
        sel_stype = st.selectbox("Type", ["All"] + present, index=0, label_visibility="collapsed")

        st.markdown('<div class="sh">Parameters</div>', unsafe_allow_html=True)
        prox = st.slider("Proximity threshold (nm)", 0.1, 2.0, 0.3, 0.1, help="Vessels closer than this are flagged")
        min_sc = st.slider("Min anomaly score", 0.0, 1.0, 0.0, 0.05, help="Lower = more sensitive")

        st.markdown('<div class="sh">Search</div>', unsafe_allow_html=True)
        q = st.text_input("MMSI or name", placeholder="e.g. 247489500", label_visibility="collapsed")

        st.markdown('<div class="sh" style="margin-top:1.3rem">Data</div>', unsafe_allow_html=True)
        st.markdown(f'''<div style="font-size:.8rem;color:#8892a4;line-height:1.9">
            <div>Messages  <span style="color:#e2e8f0;font-family:JetBrains Mono,monospace">{n_msgs:,}</span></div>
            <div>Vessels  <span style="color:#e2e8f0;font-family:JetBrains Mono,monospace">{n_vessels:,}</span></div>
            <div>Algorithms  <span style="color:#e2e8f0;font-family:JetBrains Mono,monospace">{len(sel_types)}</span></div>
        </div>''', unsafe_allow_html=True)

    all_a, summary = run_detections(pos, prox)
    adf = pd.DataFrame(all_a) if all_a else pd.DataFrame(
        columns=["mmsi", "ship_name", "anomaly_type", "severity", "latitude", "longitude", "speed", "detail", "timestamp", "score"])

    f = adf.copy()
    if sel_types:  f = f[f["anomaly_type"].isin(sel_types)]
    if sel_sev:    f = f[f["severity"].isin(sel_sev)]
    if sel_stype != "All":
        tnums = [n for rng, _ in SHIP_TYPE_RANGES for n in rng if get_ship_type_label(n) == sel_stype]
        if tnums:
            f = f[f["mmsi"].isin(pos[pos["type"].isin(tnums)]["MMSI"].unique())]
    if min_sc > 0: f = f[f["score"] >= min_sc]
    if q:
        ql = q.lower()
        f = f[f["mmsi"].astype(str).str.contains(ql) | f["ship_name"].str.lower().str.contains(ql, na=False) | f["detail"].str.lower().str.contains(ql, na=False)]
    f = f.sort_values("score", ascending=False).reset_index(drop=True)

    n_anom = len(f)
    n_crit = len(f[f["severity"] == "CRITICAL"])
    n_high = len(f[f["severity"] == "HIGH"])
    n_fv = f["mmsi"].nunique()
    cov = min(100, int(n_fv / max(n_vessels, 1) * 500))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card(n_anom, "Anomalies Detected", "r", "#ff2d55")
    with c2: metric_card(n_crit, "Critical Alerts", "o", "#ff2d55")
    with c3: metric_card(n_high, "High Priority", "y", "#ff9500")
    with c4: metric_card(n_fv, "Flagged Vessels", "g", "#5ac8fa")
    with c5: metric_card(f"{cov}%", "Scan Coverage", "b", "#30d158")

    tmap, tana, tlog = st.tabs(["🗺️  Maritime Map", "📊  Analysis", "📋  Anomaly Log"])

    with tmap:
        st.markdown('<div class="pnl" style="padding:.4rem">', unsafe_allow_html=True)
        st.plotly_chart(chart_map(pos, f), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown('<div class="sh">Severity Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_severity_bars(f), use_container_width=True, config={"displayModeBar": False})
        with r2:
            st.markdown('<div class="sh">Speed Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_speed_hist(pos, f), use_container_width=True, config={"displayModeBar": False})
        with r3:
            st.markdown('<div class="sh">Anomalies by Ship Type</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_type_pie(f, pos), use_container_width=True, config={"displayModeBar": False})

    with tana:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown('<div class="pnl">', unsafe_allow_html=True)
            st.markdown('<div class="sh">Detection Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_radar(summary, sel_types), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="pnl">', unsafe_allow_html=True)
            algo_status_panel(summary, sel_types)
            st.markdown('</div>', unsafe_allow_html=True)
        with a2:
            st.markdown('<div class="pnl">', unsafe_allow_html=True)
            st.markdown('<div class="sh">Detections Per Algorithm</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_algo_hbar(summary), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="pnl">', unsafe_allow_html=True)
            st.markdown('<div class="sh">Top Flagged Vessels</div>', unsafe_allow_html=True)
            top_vessels(f)
            st.markdown('</div>', unsafe_allow_html=True)

    with tlog:
        st.markdown(f'<div class="sh" style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span>Anomaly Log</span><span style="font-weight:400;color:#555;font-size:.72rem">{len(f)} records</span></div>',
                    unsafe_allow_html=True)
        anomaly_table(f)
        if len(f) > 0:
            st.download_button("📥  Export Anomalies CSV", f.to_csv(index=False),
                               file_name=f"nexus_anomalies_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")


if __name__ == "__main__":
    main()