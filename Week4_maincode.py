import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Orion Vessel Anomaly Explorer", layout="wide")
st.title("Orion Vessel Anomaly Explorer")
st.caption("Flags vessels with unusual AIS behavior in this 30-minute window.")
@st.cache_data
def load_results():
    return pd.read_csv('day4_results.csv')

results = load_results()
def status_label(row):
    if row['low_confidence']:
        return 'Low confidence'
    return 'Anomalous' if row['is_anomalous'] == True else 'Normal'

results['status'] = results.apply(status_label, axis=1)
m = folium.Map(location=[results['lat'].mean(), results['lon'].mean()], zoom_start=4)
color_map = {'Anomalous': 'red', 'Normal': 'blue', 'Low confidence': 'gray'}
for _, row in results.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=7 if row['status'] == 'Anomalous' else 4,
        color=color_map[row['status']], fill=True, fill_opacity=0.85,
        tooltip=row['Ship Name'] if pd.notna(row['Ship Name']) else str(row['MMSI']),
    ).add_to(m)
st_folium(m, width=900, height=600)

