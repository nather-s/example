import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# ============================================================
# 1. PAGE CONFIG + TITLE  (Day 1)
# ============================================================
st.set_page_config(page_title="Orion Vessel Anomaly Explorer", layout="wide")
st.title("Orion Vessel Anomaly Explorer")
st.caption("Flags vessels with unusual AIS behavior in this 30-minute window.")

# ============================================================
# 2. FUNCTIONS  -- define ALL of these before anything calls them
# ============================================================
@st.cache_data
def load_results():
    return pd.read_csv('full_anomaly_results.csv')

def status_label(row):
    if row['low_confidence']:
        return 'Low confidence'
    return 'Anomalous' if row['is_anomalous'] == True else 'Normal'

def human_readable_flags(flags_str):          # Day 2 "Show the why"
    if not isinstance(flags_str, str) or not flags_str:
        return []
    readable = []
    for flag in flags_str.split('; '):
        if 'signal gap' in flag:
            secs = int(''.join(c for c in flag.split('of')[1] if c.isdigit()))
            readable.append(f"Went silent for about {secs // 60} minutes before reappearing.")
        elif 'sharp turn' in flag:
            readable.append("Made an unusually sharp turn while underway.")
        elif 'long stop' in flag:
            secs = int(''.join(c for c in flag.split('of')[1] if c.isdigit()))
            readable.append(f"Stopped moving for about {secs // 60} minutes.")
        elif 'implied speed' in flag:
            readable.append("Position jumped in a way that's physically implausible for a ship this size.")
        else:
            readable.append(flag)
    return readable

# ============================================================
# 3. LOAD DATA  (Day 1)
# ============================================================
results = load_results()
results['status'] = results.apply(status_label, axis=1)

# ============================================================
# 4. SIDEBAR FILTERS  (Day 2 "Add controls")
#    Must come BEFORE the map, since the map needs to reflect these.
# ============================================================
st.sidebar.header("Filters")
status_filter = st.sidebar.multiselect("Status", ["Anomalous", "Normal", "Low confidence"],
                                         default=["Anomalous", "Normal"])
min_score = st.sidebar.slider("Minimum score (anomalous only)", 0.0, 4.0, 0.0, 0.1)
name_search = st.sidebar.text_input("Search by ship name or MMSI")

filtered = results[results['status'].isin(status_filter)]
filtered = filtered[(filtered['score'].fillna(0) >= min_score) | (filtered['status'] != 'Anomalous')]
if name_search:
    mask = (filtered['Ship Name'].fillna('').str.contains(name_search, case=False) |
            filtered['MMSI'].astype(str).str.contains(name_search))
    filtered = filtered[mask]

# ============================================================
# 5. EMPTY STATE  (Day 2 "Handle the empty cases")
#    Must come AFTER filtering, BEFORE the map/selectbox below --
#    otherwise an empty `filtered` crashes the next section.
# ============================================================
if filtered.empty:
    st.warning("No tracks match these filters. Try widening your filter selection.")
    st.stop()   # nothing below this line runs if we hit this

# ============================================================
# 6. MAP + DETAIL PANEL
#    Day 1 "Plot the right things" + Day 2 "Link map and detail"
#    Loops over `filtered`, not `results` -- this is what makes the
#    filters above actually change what's shown.
# ============================================================
color_map = {'Anomalous': 'red', 'Normal': 'blue', 'Low confidence': 'gray'}
m = folium.Map(location=[filtered['lat'].mean(), filtered['lon'].mean()], zoom_start=4)
for _, row in filtered.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=7 if row['status'] == 'Anomalous' else 4,
        color=color_map[row['status']], fill=True, fill_opacity=0.85,
        tooltip=row['Ship Name'] if pd.notna(row['Ship Name']) else str(row['MMSI']),
    ).add_to(m)

col1, col2 = st.columns([2, 1])

with col1:
    st_folium(m, width=700, height=600)

with col2:
    st.subheader("Select a track")
    options = filtered.apply(
        lambda r: f"{r['Ship Name'] if pd.notna(r['Ship Name']) else 'Unknown'} (MMSI {r['MMSI']})", axis=1
    ).tolist()
    selected = st.selectbox("Track", options)
    sel_row = filtered.iloc[options.index(selected)]

    st.markdown(f"### {sel_row['Ship Name'] if pd.notna(sel_row['Ship Name']) else 'Unknown vessel'}")
    if sel_row['status'] == 'Low confidence':
        st.info(f"Only {int(sel_row['n_pings'])} ping(s) — not enough data to assess.")
    elif sel_row['status'] == 'Anomalous':
        st.error("Flagged as unusual:")
        for reason in human_readable_flags(sel_row['flags']):
            st.write(f"- {reason}")
    else:
        st.success("No unusual behavior detected.")

# ============================================================
# 7. LEGEND
