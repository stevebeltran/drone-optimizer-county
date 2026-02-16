import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import itertools

# --- PAGE CONFIG ---
st.set_page_config(page_title="Drone Logistics Portal", layout="wide", page_icon="🛰️")

# --- 1. INITIALIZE SESSION STATE ---
if 'box_open' not in st.session_state:
    st.session_state.box_open = True

# --- 2. HEADER ---
if st.sidebar.button("🔄 Reset & Upload New Data"):
    st.session_state.box_open = True
    st.rerun()

st.title("🛰️ Strategic Drone Deployment Optimizer")
st.markdown("#### **Geospatial Operations Analysis Tool**")

# --- MISSOURI FIPS LOOKUP ---
FIPS_MAP = {
    '183': 'St. Charles County',
    '189': 'St. Louis County',
    '510': 'St. Louis City',
    '099': 'Jefferson County',
    '071': 'Franklin County',
    '113': 'Lincoln County',
    '219': 'Warren County',
    '019': 'Boone County',
    '095': 'Jackson County',
    '037': 'Cass County',
    '047': 'Clay County',
    '165': 'Platte County',
    '077': 'Greene County'
}

# --- SPEED OPTIMIZATION: CACHING ---
@st.cache_data
def load_shapefile(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None: gdf.set_crs(epsg=4269, inplace=True)
    gdf['geometry'] = gdf['geometry'].simplify(0.0001, preserve_topology=True)
    return gdf

def filter_geo_data(gdf, county_fips, district_name):
    if county_fips and county_fips != "ALL":
        if 'COUNTYFP' in gdf.columns:
            gdf = gdf[gdf['COUNTYFP'] == county_fips]
        elif 'CNTYID' in gdf.columns:
            gdf = gdf[gdf['CNTYID'] == county_fips]

    dist_col = next((c for c in ['NAMELSAD', 'NAME', 'DISTRICT'] if c in gdf.columns), gdf.columns[0])
    
    if district_name == "SHOW ALL IN SELECTION":
        active_gdf = gdf.to_crs(epsg=4326)
        boundary = unary_union(active_gdf.geometry)
    else:
        active_gdf = gdf[gdf[dist_col] == district_name].to_crs(epsg=4326)
        boundary = active_gdf.iloc[0].geometry

    return gdf, active_gdf, boundary

# --- 3. DATA IMPORT ---
call_data, station_data, shot_data, shape_components = None, None, None, []

with st.expander("📁 Secure Data Import", expanded=st.session_state.box_open):
    uploaded_files = st.file_uploader("Upload Incident CSVs, 'shots.csv', and Shapefiles", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        fname = f.name.lower()
        if fname == "calls.csv": call_data = f
        elif fname == "stations.csv": station_data = f
        elif fname == "shots.csv": shot_data = f
        elif any(fname.endswith(ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
            shape_components.append(f)

    if call_data and station_data and len(shape_components) >= 3:
        if st.session_state.box_open:
            st.session_state.box_open = False
            st.rerun()

STATION_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#800000", "#333333", "#000075"]

# --- 4. MAIN ANALYSIS ENGINE ---
if call_data and station_data and len(shape_components) >= 3:
    if not os.path.exists("temp"): os.mkdir("temp")
    for f in shape_components:
        with open(os.path.join("temp", f.name), "wb") as buffer:
            buffer.write(f.getbuffer())
    
    try:
        shp_path = [os.path.join("temp", f.name) for f in shape_components if f.name.endswith('.shp')][0]
        raw_gdf = load_shapefile(shp_path)
        
        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        
        county_id_col = next((c for c in ['COUNTYFP', 'CNTYID', 'FP_COUNTY'] if c in raw_gdf.columns), None)
        selected_county_fips = "ALL"
        
        if county_id_col:
            unique_fips = sorted(raw_gdf[county_id_col].astype(str).unique().tolist())
            def format_county_name(fips_code):
                if fips_code == "ALL": return "ALL COUNTIES"
                return FIPS_MAP.get(fips_code, f"County {fips_code}")

            selected_county_fips = ctrl_col1.selectbox("🏛️ Filter by County", ["ALL"] + unique_fips, format_func=format_county_name)
            
            if selected_county_fips != "ALL":
                filtered_view = raw_gdf[raw_gdf[county_id_col].astype(str) == selected_county_fips]
            else:
                filtered_view = raw_gdf
        else:
            filtered_view = raw_gdf

        dist_col_name = next((c for c in ['NAMELSAD', 'NAME', 'DISTRICT'] if c in filtered_view.columns), None)
        if dist_col_name:
            dist_options = ["SHOW ALL IN SELECTION"] + sorted(filtered_view[dist_col_name].astype(str).unique().tolist())
            dist_choice = ctrl_col1.selectbox(f"📍 Active Jurisdiction ({dist_col_name})", dist_options)
        else:
            dist_choice = "SHOW ALL IN SELECTION"

        gdf_all, active_gdf, city_boundary = filter_geo_data(raw_gdf, selected_county_fips, dist_choice)
        utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
        epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
        city_m = active_gdf.to_crs(epsg=epsg_code).unary_union
        
        df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
        df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
        calls_in_city = gdf_calls[gdf_calls.within(city_boundary)].to_crs(epsg=epsg_code)
        calls_in_city['point_idx'] = range(len(calls_in_city))
        
        # PRE-CALC
        radius_m = 3218.69 
        station_metadata = []
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            mask = calls_in_city.geometry.distance(s_pt_m) <= radius_m
            indices = set(calls_in_city[mask]['point_idx'])
            clipped_buf = s_pt_m.buffer(radius_m).intersection(city_m)
            station_metadata.append({'name': row['name'], 'lat': row['lat'], 'lon': row['lon'], 'clipped_m': clipped_buf, 'indices': indices})

        # OPTIMIZER
        st.sidebar.header("🎯 Optimizer Controls")
        k = st.sidebar.slider("Drones to Deploy", 0, len(station_metadata), min(2, len(station_metadata)))
        strategy = st.sidebar.radio("Optimization Goal", ("Maximize Call Volume", "Maximize Land Equity"))

        if k == 0:
            active_names = []
            st.sidebar.info("🧪 **Generation Mode**: Enable 'Suggested Sites' below.")
        else:
            combos = list(itertools.combinations(range(len(station_metadata)), k))
            if len(combos) > 500: combos = combos[:500] 
            
            best_call_combo, max_calls = None, -1
            best_geo_combo, max_area = -1, -1
            
            for combo in combos:
                u_set = set().union(*(station_metadata[i]['indices'] for i in combo))
                if len(u_set) > max_calls: max_calls = len(u_set); best_call_combo = combo
                if strategy == "Maximize Land Equity":
                    u_geo = unary_union([station_metadata[i]['clipped_m'] for i in combo])
                    if u_geo.area > max_area: max_area = u_geo.area; best_geo_combo = combo
            
            default_sel = [station_metadata[i]['name'] for i in (best_call_combo if strategy == "Maximize Call Volume" else (best_geo_combo if best_geo_combo != -1 else best_call_combo))]
            active_names = ctrl_col2.multiselect("📡 Current Drone List", options=df_stations_all['name'].tolist(), default=default_sel)
        
        # LAYER CONTROLS
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Layer Controls")
        
        show_shots = False
        df_shots = None
        if shot_data:
            show_shots = st.sidebar.toggle("Show Shot Detection Events", value=False)
            if show_shots:
                try: df_shots = pd.read_csv(shot_data)
                except: pass
        
        show_suggestions = st.sidebar.toggle("Show Suggested Coverage Sites", value=False)
        # --- NEW TOGGLE ---
        show_health = st.sidebar.toggle("Show Health Score Banner", value=True)

        # --- METRICS ---
        active_data = [s for s in station_metadata if s['name'] in active_names]
        active_indices = [s['indices'] for s in active_data]
        all_ids = set().union(*active_indices) if active_indices else set()
        cap_perc = (len(all_ids) / len(calls_in_city)) * 100 if len(calls_in_city) > 0 else 0
        total_union_geo = unary_union([s['clipped_m'] for s in active_data]) if active_data else None
        land_perc = (total_union_geo.area / city_m.area * 100) if total_union_geo else 0
        
        overlap_perc = 0.0
        if len(active_data) > 1:
            active_bufs = [s['clipped_m'] for s in active_data]
            inters = [active_bufs[i].intersection(active_bufs[j]) for i in range(len(active_bufs)) for j in range(i+1, len(active_bufs)) if not active_bufs[i].intersection(active_bufs[j]).is_empty]
            overlap_perc = (unary_union(inters).area / city_m.area * 100) if inters else 0.0

        suggested_coords = []
        if land_perc < 99.0:
            current_covered = total_union_geo if total_union_geo else Polygon()
            uncovered_poly = city_m.difference(current_covered)
            max_iterations = 25 
            for _ in range(max_iterations):
                if uncovered_poly.is_empty or (uncovered_poly.area / city_m.area) < 0.01: break 
                if isinstance(uncovered_poly, MultiPolygon):
                    valid_geoms = [g for g in uncovered_poly.geoms if g.area > 1000] 
                    if not valid_geoms: break
                    target_chunk = max(valid_geoms, key=lambda g: g.area)
                else:
                    target_chunk = uncovered_poly
                new_site_pt = target_chunk.representative_point()
                p_geo = gpd.GeoSeries([new_site_pt], crs=epsg_code).to_crs(epsg=4326).iloc[0]
                suggested_coords.append({'lat': p_geo.y, 'lon': p_geo.x})
                new_coverage = new_site_pt.buffer(radius_m)
                uncovered_poly = uncovered_poly.difference(new_coverage)

        # HEALTH SCORE DISPLAY LOGIC
        if show_health:
            norm_redundancy = min(overlap_perc / 39.0, 1.0) * 100
            health_score = (cap_perc * 0.50) + (land_perc * 0.25) + (norm_redundancy * 0.25)

            if health_score >= 85: h_color, h_label = "#28a745", "OPTIMAL"
            elif health_score >= 75: h_color, h_label = "#94c11f", "SUFFICIENT"
            elif health_score >= 55: h_color, h_label = "#ffc107", "MARGINAL"
            else: h_color, h_label = "#dc3545", "CRITICAL"

            st.markdown(f"""
                <div style="background-color: {h_color}; padding: 10px; border-radius: 5px; color: white; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 1.2em; font-weight: bold;">Department Health Score: {health_score:.1f}%</span>
                    <span style="font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 2px 10px; border-radius: 4px;">{h_label}</span>
                </div>
                """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Calls", f"{len(calls_in_city):,}")
        m2.metric("Response Capacity", f"{cap_perc:.1f}%")
        m3.metric("Land Covered", f"{land_perc:.1f}%")
        m4.metric("Redundancy", f"{overlap_perc:.1f}%")
        m5.metric("Uncovered Calls", f"{len(calls_in_city) - len(all_ids):,}")

        # SIDEBAR SCORECARD
        st.sidebar.markdown("---")
        with st.sidebar.expander("📝 Tactical Scorecard", expanded=True):
            c_disp = FIPS_MAP.get(selected_county_fips, selected_county_fips) if selected_county_fips != "ALL" else "All Counties"
            
            # Conditionally add Health Score to text report
            health_text = f"Status: {h_label}\n" if show_health else ""
            
            summary_text = f"""DRONE DEPLOYMENT ANALYSIS
---------------------------------
County: {c_disp}
Jurisdiction: {dist_choice}
Total Calls: {len(calls_in_city):,}
Coverage: {land_perc:.1f}%
{health_text}
DEPLOYED ASSETS:
""" + "\n".join([f"✅ {name}" for name in active_names])
            if suggested_coords and show_suggestions:
                summary_text += "\n\n⚠️ SUGGESTED EXPANSIONS:\n"
                for i, c in enumerate(suggested_coords):
                    summary_text += f"📍 Site {i+1}: {c['lat']:.5f}, {c['lon']:.5f}\n"
            st.text_area("Copy Report:", summary_text, height=300)

        if suggested_coords and show_suggestions:
            st.sidebar.markdown("### 💡 Recommended Sites")
            st.sidebar.info(f"Generated {len(suggested_coords)} sites.")
            for i, c in enumerate(suggested_coords):
                st.sidebar.code(f"{c['lat']:.5f}, {c['lon']:.5f}", language="text")

        # MAP
        fig = go.Figure()
        
        for _, row in gdf_all.to_crs(epsg=4326).iterrows():
            geom = row.geometry
            p_list = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
            for p in p_list:
                bx, by = p.exterior.coords.xy
                fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#444", width=1), showlegend=False, hoverinfo='skip'))
        
        sample = calls_in_city.to_crs(epsg=4326).sample(min(2000, len(calls_in_city)))
        fig.add_trace(go.Scattermap(
            lat=sample.geometry.y, 
            lon=sample.geometry.x, 
            mode='markers', 
            marker=dict(size=4, color='#000080', opacity=0.3), 
            name="Incidents",
            hoverinfo='skip'
        ))

        if show_shots and df_shots is not None:
             fig.add_trace(go.Scattermap(
                lat=df_shots['lat'],
                lon=df_shots['lon'],
                mode='markers',
                marker=dict(symbol='triangle', size=10, color='#FF4500', opacity=0.9),
                name="Shot Detection",
                text=df_shots['point_id'] if 'point_id' in df_shots.columns else None,
                hoverinfo='text+lat+lon'
            ))

        if show_suggestions:
            for i, c in enumerate(suggested_coords):
                angles = np.linspace(0, 2*np.pi, 100)
                clats = c['lat'] + (2/69.172) * np.sin(angles)
                clons = c['lon'] + (2/(69.172 * np.cos(np.radians(c['lat'])))) * np.cos(angles)
                fig.add_trace(go.Scattermap(
                    lat=list(clats), lon=list(clons), 
                    mode='markers', marker=dict(size=4, color='#FF00FF'), 
                    name=f"Proposed Coverage {i+1}", hoverinfo='skip', showlegend=False
                ))
                fig.add_trace(go.Scattermap(
                    lat=[c['lat']], lon=[c['lon']], 
                    mode='markers+text', marker=dict(size=12, color='#FF00FF', symbol='circle'), 
                    text=[f"NEW SITE {i+1}"], textposition="top center", 
                    name=f"Suggestion {i+1}", hoverinfo='text'
                ))
        
        all_st_names = df_stations_all['name'].tolist()
        for s in active_data:
            color = STATION_COLORS[all_st_names.index(s['name']) % len(STATION_COLORS)]
            angles = np.linspace(0, 2*np.pi, 60)
            clats = s['lat'] + (2/69.172) * np.sin(angles)
            clons = s['lon'] + (2/(69.172 * np.cos(np.radians(s['lat'])))) * np.cos(angles)
            fig.add_trace(go.Scattermap(
                lat=list(clats) + [clats[0]], lon=list(clons) + [clons[0]], 
                mode='lines', line=dict(color=color, width=4.5), 
                hoverinfo='skip', showlegend=False
            ))
            fig.add_trace(go.Scattermap(
                lat=[s['lat']], lon=[s['lon']], 
                mode='markers', marker=dict(size=12, color=color), 
                name=s['name'], hoverinfo='name'
            ))

        fig.update_layout(map_style="open-street-map", map_zoom=11, map_center={"lat": city_boundary.centroid.y, "lon": city_boundary.centroid.x}, margin={"r":0,"t":0,"l":0,"b":0}, height=750)
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("System Ready: Please upload deployment files above to initialize session.")
