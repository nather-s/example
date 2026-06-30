import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import anomaly_detecor as ad

# ============================================================
# PAGE CONFIG + TITLE
# ============================================================
st.set_page_config(page_title="Orion Anomaly Detector", layout="wide")
st.title("🚢 Orion Anomaly Detector")
st.caption("AIS vessel anomaly detection — movement behavior and identity/registration checks.")

# ============================================================
# SHARED FUNCTIONS
# ============================================================
@st.cache_data
def load_results():
    return pd.read_csv('full_anomaly_results.csv')

@st.cache_data
def load_raw_positions():
    df = pd.read_csv('formatted_ship_messages.csv')
    POSITION_TYPES = ['PositionReport', 'StandardClassBPositionReport', 'ExtendedClassBPositionReport']
    positions = df[df['Message Type'].isin(POSITION_TYPES)].copy()
    positions['time'] = pd.to_datetime(positions['Received UTC'], format='%H:%M:%S')
    return positions.sort_values(['MMSI', 'time']).reset_index(drop=True)

@st.cache_data
def load_raw_df():
    df = pd.read_csv('formatted_ship_messages.csv')
    df['time'] = pd.to_datetime(df['Received UTC'], format='%H:%M:%S')
    return df

@st.cache_data
def score_at_cutoff(cutoff_str):
    """Cached per cutoff -- revisiting a time you've already scored is instant."""
    raw_df = load_raw_df()
    full_positions = load_raw_positions()
    return ad.score_as_of(cutoff_str, raw_df, full_positions)

def overall_status(row):
    if row['is_anomalous'] == True:
        return 'Movement anomaly'
    if row['identity_flagged']:
        return 'Identity flag'
    if row['low_confidence']:
        return 'Low confidence'
    return 'Normal'

def human_readable_flags(flags_str):
    if not isinstance(flags_str, str) or not flags_str:
        return []
    readable = []
    for flag in flags_str.split('; '):
        if 'signal gap' in flag:
            secs = int(''.join(c for c in flag.split('of')[1] if c.isdigit()))
            readable.append(f"Went silent for about {secs // 60} minutes before reappearing.")
        elif 'sharp turn' in flag:
            readable.append("Made an unusually sharp turn while underway, away from any port.")
        elif 'long stop' in flag:
            secs = int(''.join(c for c in flag.split('of')[1] if c.isdigit()))
            readable.append(f"Stopped moving for about {secs // 60} minutes.")
        elif 'implied speed' in flag:
            readable.append("Position jumped in a way that's physically implausible for a ship this size.")
        else:
            readable.append(flag)
    return readable

def human_readable_identity(flags_str):
    if not isinstance(flags_str, str) or not flags_str:
        return []
    return flags_str.split('; ')

def render_explorer(results, raw_positions, key_prefix):
    """Shared rendering for both tabs. key_prefix keeps widget keys unique
    so the same controls can exist in two tabs without colliding."""
    results = results.copy()
    results['status'] = results.apply(overall_status, axis=1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total vessels", f"{len(results):,}")
    m2.metric("Scoreable (movement)", f"{(~results['low_confidence']).sum():,}")
    m3.metric("🔴 Movement anomalies", f"{(results['status']=='Movement anomaly').sum():,}")
    m4.metric("🟠 Identity/registration flags", f"{results['identity_flagged'].sum():,}")
    st.markdown("---")

    st.sidebar.header(f"Filters ({key_prefix})")
    status_filter = st.sidebar.multiselect(
        "Status", ["Movement anomaly", "Identity flag", "Normal", "Low confidence"],
        default=["Movement anomaly", "Identity flag", "Normal"], key=f"{key_prefix}_status",
    )
    min_score = st.sidebar.slider("Minimum movement score", 0.0, 4.0, 0.0, 0.1, key=f"{key_prefix}_minscore")
    name_search = st.sidebar.text_input("Search by ship name or MMSI", key=f"{key_prefix}_search")

    filtered = results[results['status'].isin(status_filter)]
    filtered = filtered[(filtered['score'].fillna(0) >= min_score) | (filtered['status'] != 'Movement anomaly')]
    if name_search:
        mask = (filtered['Ship Name'].fillna('').str.contains(name_search, case=False) |
                filtered['MMSI'].astype(str).str.contains(name_search))
        filtered = filtered[mask]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "🔴 Movement anomaly — unusual behavior while underway\n\n"
        "🟠 Identity flag — registration/data-integrity issue\n\n"
        "🔵 Normal\n\n"
        "⚪ Low confidence — too few pings to assess movement"
    )

    if filtered.empty:
        st.warning("No tracks match these filters. Try widening your filter selection.")
        return

    col1, col2 = st.columns([2, 1])
    with col2:
        st.subheader("Drill down into a vessel")
        options = filtered.apply(
            lambda r: f"{r['Ship Name'] if pd.notna(r['Ship Name']) else 'Unknown'} (MMSI {r['MMSI']})", axis=1
        ).tolist()
        selected = st.selectbox("Select track", options, key=f"{key_prefix}_select")
        sel_row = filtered.iloc[options.index(selected)]

        st.markdown(f"### {sel_row['Ship Name'] if pd.notna(sel_row['Ship Name']) else 'Unknown vessel'}")
        st.caption(f"MMSI {sel_row['MMSI']}")

        if sel_row['status'] == 'Movement anomaly':
            st.error(f"Movement anomaly (score {sel_row['score']})")
            for reason in human_readable_flags(sel_row['flags']):
                st.write(f"- {reason}")
        elif sel_row['status'] == 'Identity flag':
            st.warning("Identity / registration flag")
            for reason in human_readable_identity(sel_row['identity_flags']):
                st.write(f"- {reason}")
        elif sel_row['status'] == 'Low confidence':
            st.info(f"Only {int(sel_row['n_pings'])} ping(s) so far — not enough movement data to assess.")
        else:
            st.success("No unusual behavior or identity issues detected.")
            if bool(sel_row.get('near_port', False)):
                st.caption("ℹ️ Near a busy port — sharp turns there are treated as routine, not flagged.")

        st.markdown("---")
        sel_track = raw_positions[raw_positions['MMSI'] == sel_row['MMSI']]
        s1, s2, s3 = st.columns(3)
        s1.metric("Pings so far", len(sel_track))
        if len(sel_track) >= 2:
            speeds = sel_track['Speed Over Ground (knots)']
            s2.metric("Avg speed (kt)", f"{speeds.mean():.1f}")
            s3.metric("Max speed (kt)", f"{speeds.max():.1f}")
            st.markdown("**Speed over time**")
            st.line_chart(sel_track.set_index('time')[['Speed Over Ground (knots)']])

    color_map = {'Movement anomaly': 'red', 'Identity flag': 'orange', 'Normal': 'blue', 'Low confidence': 'gray'}
    m = folium.Map(location=[filtered['lat'].mean(), filtered['lon'].mean()], zoom_start=4)
    for _, row in filtered.iterrows():
        if row['MMSI'] == sel_row['MMSI']:
            continue
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5 if row['status'] in ('Movement anomaly', 'Identity flag') else 4,
            color=color_map[row['status']], fill=True, fill_opacity=0.4,
            tooltip=row['Ship Name'] if pd.notna(row['Ship Name']) else str(row['MMSI']),
        ).add_to(m)

    track_points = list(zip(sel_track['Latitude'], sel_track['Longitude']))
    if len(track_points) >= 2:
        folium.PolyLine(track_points, color=color_map[sel_row['status']], weight=4, opacity=0.9).add_to(m)
    for pt in track_points:
        folium.CircleMarker(location=pt, radius=3, color=color_map[sel_row['status']], fill=True).add_to(m)
    if track_points:
        m.fit_bounds(track_points)

    with col1:
        st.subheader("Positions — click a marker for its path and reasoning")
        st_folium(m, width=700, height=550, key=f"{key_prefix}_map")

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["📊 Snapshot Explorer", "⏱️ Live Replay"])

with tab1:
    st.caption("Full 30-minute window, fully scored.")
    results = load_results()
    raw_positions = load_raw_positions()
    render_explorer(results, raw_positions, key_prefix="snap")

with tab2:
    st.caption(
        "Replays the window as if messages were arriving live. Vessels start as "
        "'low confidence' and graduate to a real verdict once enough data has arrived — "
        "watch MOBY LEGACY (MMSI 247484300) flip to anomalous partway through."
    )
    start_str, end_str = ad.get_time_bounds('formatted_ship_messages.csv')
    step_seconds = 30
    steps = pd.date_range(
        pd.to_datetime(start_str, format='%H:%M:%S'),
        pd.to_datetime(end_str, format='%H:%M:%S'),
        freq=f'{step_seconds}s',
    )
    step_options = [t.strftime('%H:%M:%S') for t in steps]

    if 'replay_index' not in st.session_state:
        st.session_state.replay_index = 0

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        idx = st.slider("Replay up to time:", 0, len(step_options) - 1,
                         value=st.session_state.replay_index, key="replay_slider")
        st.session_state.replay_index = idx
    with c2:
        if st.button("⏮ Step back") and st.session_state.replay_index > 0:
            st.session_state.replay_index -= 1
            st.rerun()
    with c3:
        if st.button("Step forward ⏭") and st.session_state.replay_index < len(step_options) - 1:
            st.session_state.replay_index += 1
            st.rerun()

    cutoff_str = step_options[st.session_state.replay_index]
    st.markdown(f"**Showing data received through `{cutoff_str}`**")

    with st.spinner(f"Scoring all data through {cutoff_str}..."):
        live_results = score_at_cutoff(cutoff_str)

    if live_results.empty:
        st.info("No messages received yet at this point in the replay.")
    else:
        live_positions = load_raw_positions()
        live_positions = live_positions[live_positions['time'] <= pd.to_datetime(cutoff_str, format='%H:%M:%S')]
        render_explorer(live_results, live_positions, key_prefix="live")