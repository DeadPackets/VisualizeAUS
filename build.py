"""
Build script for VisualizeAUS static site.
Reads aus_courses.db, generates interactive Plotly charts, and assembles
a complete static HTML site in the _site/ directory.

Usage: python build.py
"""

import sqlite3
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import networkx as nx

import analysis

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = "aus_courses.db"
OUT_DIR = Path("_site")
TEMPLATE = "plotly_white"
AUS_GOLD = "#C4972F"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

chart_specs = {}  # chart_id -> {data, layout}; embedded once for lazy rendering


def chart_html(fig, chart_id):
    """Register a figure for lazy rendering and return a sized placeholder div.

    Charts are no longer rendered eagerly on load (62 simultaneous Plotly
    layouts is punishing on phones). Each figure's spec is stored in
    ``chart_specs`` — embedded once as a single JSON blob — and an
    IntersectionObserver calls ``Plotly.newPlot`` only when the placeholder
    scrolls near the viewport. The placeholder reserves the chart's height so
    deferring the render causes no layout shift.
    """
    chart_specs[chart_id] = json.loads(pio.to_json(fig))
    height = fig.layout.height or 450
    return (f'<div id="chart-{chart_id}" class="lazy-chart" data-cid="{chart_id}" '
            f'style="min-height:{height}px;"></div>')


def parse_time_minutes(t):
    try:
        parts = t.strip().split()
        h, m = map(int, parts[0].split(":"))
        if parts[1].lower() == "pm" and h != 12:
            h += 12
        if parts[1].lower() == "am" and h == 12:
            h = 0
        return h * 60 + m
    except Exception:
        return 9999


def classify_semester(name):
    name = name.lower()
    if "fall" in name:
        return "Fall"
    if "spring" in name:
        return "Spring"
    if "wintermester" in name:
        return "Wintermester"
    if "summer ii" in name or "summer 2" in name:
        return "Summer II"
    if "summer iii" in name or "summer 3" in name:
        return "Summer III"
    return "Summer"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

conn = sqlite3.connect(DB_PATH)

# Table counts
table_counts = {}
for table in ["semesters", "subjects", "courses", "instructors", "levels",
              "attributes", "catalog", "section_details", "course_dependencies"]:
    table_counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

# ---------------------------------------------------------------------------
# Generate all charts
# ---------------------------------------------------------------------------
charts = {}

# ===== 1. University Growth =====
df_growth = pd.read_sql_query("""
    SELECT s.term_id, s.term_name,
           COUNT(*) as total_sections,
           COUNT(DISTINCT c.subject || c.course_number) as unique_courses,
           COUNT(DISTINCT c.instructor_name) as unique_instructors
    FROM courses c
    JOIN semesters s ON c.term_id = s.term_id
    GROUP BY c.term_id ORDER BY c.term_id
""", conn)
df_growth["semester_type"] = df_growth["term_name"].apply(classify_semester)
df_regular = df_growth[df_growth["semester_type"].isin(["Fall", "Spring"])].copy()

fig = go.Figure()
color_map = {"Fall": "#c0392b", "Spring": "#2471a3", "Summer": "#27ae60",
             "Summer II": "#27ae60", "Summer III": "#27ae60", "Wintermester": "#8e44ad"}
# Sort chronologically by term_id so x-axis is in order
df_growth_sorted = df_growth.sort_values("term_id")
for sem_type in ["Fall", "Spring", "Summer", "Summer II", "Summer III", "Wintermester"]:
    mask = df_growth_sorted["semester_type"] == sem_type
    if not mask.any():
        continue
    fig.add_trace(go.Scatter(
        x=df_growth_sorted.loc[mask, "term_name"], y=df_growth_sorted.loc[mask, "total_sections"],
        mode="markers", name=sem_type,
        marker=dict(color=color_map.get(sem_type, "#95a5a6"), size=8),
        hovertemplate="%{x}<br>%{y} sections<extra></extra>"))
z = np.polyfit(range(len(df_regular)), df_regular["total_sections"].values, 1)
p = np.poly1d(z)
fig.add_trace(go.Scatter(
    x=df_regular["term_name"], y=p(range(len(df_regular))),
    mode="lines", name=f"Trend (+{z[0]:.1f}/sem)",
    line=dict(color="rgba(0,0,0,0.3)", dash="dash", width=2)))
fig.update_layout(template=TEMPLATE, title="Course Sections Per Semester (2005-2026)",
                  xaxis_title="Semester", yaxis_title="Number of Sections", height=500,
                  xaxis=dict(tickangle=-45, categoryorder="array",
                             categoryarray=df_growth_sorted["term_name"].tolist(), dtick=4),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts["growth"] = chart_html(fig, "growth")

# Growth stats
growth_pct = (df_regular["total_sections"].iloc[-1] / df_regular["total_sections"].iloc[0] - 1) * 100
peak_sem = df_regular.loc[df_regular["total_sections"].idxmax(), "term_name"]
peak_val = df_regular["total_sections"].max()

# ===== 1b. Unique Courses vs Sections =====
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_regular["term_name"], y=df_regular["total_sections"],
    name="Total Sections", mode="lines+markers", line=dict(color="#3498db", width=2),
    marker=dict(size=5)), secondary_y=False)
fig.add_trace(go.Scatter(x=df_regular["term_name"], y=df_regular["unique_courses"],
    name="Unique Courses", mode="lines+markers", line=dict(color="#e74c3c", width=2),
    marker=dict(size=5)), secondary_y=True)
fig.update_layout(template=TEMPLATE, title="Total Sections vs Unique Courses", height=450,
                  xaxis=dict(tickangle=-45, dtick=4),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_yaxes(title_text="Total Sections", secondary_y=False)
fig.update_yaxes(title_text="Unique Courses", secondary_y=True)
charts["courses_vs_sections"] = chart_html(fig, "courses_vs_sections")

# ===== 2. Subject Popularity =====
df_subjects = pd.read_sql_query("""
    SELECT c.subject, sub.long_name, COUNT(*) as total_sections,
           COUNT(DISTINCT c.term_id) as semesters_active,
           COUNT(DISTINCT c.instructor_name) as instructors
    FROM courses c JOIN subjects sub ON c.subject = sub.short_name
    GROUP BY c.subject ORDER BY total_sections DESC
""", conn)
top25 = df_subjects.head(25)
fig = px.bar(top25, x="subject", y="total_sections", color="total_sections",
             color_continuous_scale="Viridis",
             hover_data=["long_name", "semesters_active", "instructors"],
             labels={"total_sections": "Total Sections", "subject": "Subject Code"},
             title="Top 25 Subjects by Total Course Sections")
fig.update_layout(template=TEMPLATE, height=500, showlegend=False)
fig.update_coloraxes(showscale=False)
charts["subjects_bar"] = chart_html(fig, "subjects_bar")

# ===== 2b. Subject Heatmap =====
df_heatmap = pd.read_sql_query("""
    SELECT c.subject, CAST(SUBSTR(s.term_name, -4) AS INTEGER) as year, COUNT(*) as sections
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    GROUP BY c.subject, year
""", conn)
top30 = df_subjects.head(30)["subject"].tolist()
df_hm = df_heatmap[df_heatmap["subject"].isin(top30)].pivot_table(
    index="subject", columns="year", values="sections", fill_value=0)
df_hm = df_hm.loc[df_hm.sum(axis=1).sort_values(ascending=True).index]
fig = px.imshow(df_hm, aspect="auto", color_continuous_scale="YlOrRd",
                title="Course Sections Heatmap: Top 30 Subjects by Year",
                labels={"x": "Year", "y": "Subject", "color": "Sections"})
fig.update_layout(template=TEMPLATE, height=700)
charts["subject_heatmap"] = chart_html(fig, "subject_heatmap")

# ===== 2c. Subject lines =====
df_subj_time = pd.read_sql_query("""
    SELECT c.subject, s.term_name, s.term_id, COUNT(*) as sections
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%'
    GROUP BY c.subject, s.term_id ORDER BY s.term_id
""", conn)
top10_subjects = df_subjects.head(10)["subject"].tolist()
df_top10 = df_subj_time[df_subj_time["subject"].isin(top10_subjects)].copy()
distinct_colors = ["#c0392b", "#2471a3", "#27ae60", "#8e44ad", "#d68910",
                   "#1abc9c", "#e74c3c", "#2c3e50", "#e67e22", "#16a085"]
fig = px.line(df_top10, x="term_name", y="sections", color="subject",
              title="Top 10 Subjects: Section Count Over Time",
              labels={"sections": "Sections", "term_name": "Semester"},
              color_discrete_sequence=distinct_colors)
fig.update_traces(line=dict(width=2.5))
fig.update_layout(template=TEMPLATE, height=550, xaxis=dict(tickangle=-45, dtick=4),
                  legend=dict(title="Subject", font=dict(size=12)))
charts["subject_lines"] = chart_html(fig, "subject_lines")

# ===== 3. Instructors =====
df_instructors = pd.read_sql_query("""
    SELECT instructor_name, COUNT(*) as total_sections,
           COUNT(DISTINCT subject) as subjects_taught,
           COUNT(DISTINCT term_id) as semesters_active,
           MIN(term_id) as first_term, MAX(term_id) as last_term
    FROM courses WHERE instructor_name != '' AND instructor_name != 'TBA'
    GROUP BY instructor_name ORDER BY total_sections DESC
""", conn)
top30_inst = df_instructors.head(30)
fig = px.bar(top30_inst.iloc[::-1], x="total_sections", y="instructor_name", orientation="h",
             color="semesters_active", color_continuous_scale=["#d4e6f1", "#1a5276"],
             hover_data=["subjects_taught", "semesters_active"],
             title="Top 30 Instructors by Total Sections Taught",
             labels={"total_sections": "Total Sections", "instructor_name": "",
                     "semesters_active": "Semesters Active"})
fig.update_layout(template=TEMPLATE, height=700,
                  coloraxis_colorbar=dict(title="Semesters<br>Active", thickness=15, len=0.5))
charts["instructors"] = chart_html(fig, "instructors")

# ===== 3b. Instructor tenure =====
fig = px.histogram(df_instructors, x="semesters_active", nbins=40,
                   title="Instructor Tenure Distribution (Semesters Active)",
                   labels={"semesters_active": "Semesters Active", "count": "Instructors"},
                   color_discrete_sequence=[AUS_GOLD])
fig.update_layout(template=TEMPLATE, height=400)
charts["tenure"] = chart_html(fig, "tenure")

# ===== 3c. Active instructors =====
df_active_inst = pd.read_sql_query("""
    SELECT s.term_name, s.term_id, COUNT(DISTINCT c.instructor_name) as active_instructors
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE c.instructor_name != '' AND c.instructor_name != 'TBA'
    AND (s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%')
    GROUP BY c.term_id ORDER BY c.term_id
""", conn)
fig = px.area(df_active_inst, x="term_name", y="active_instructors",
              title="Active Instructors Per Regular Semester",
              labels={"active_instructors": "Active Instructors", "term_name": "Semester"},
              color_discrete_sequence=[AUS_GOLD])
fig.update_layout(template=TEMPLATE, height=400, xaxis=dict(tickangle=-45, dtick=4))
charts["active_instructors"] = chart_html(fig, "active_instructors")

# ===== 3d. TBA rate =====
df_tba = pd.read_sql_query("""
    SELECT s.term_name, s.term_id,
           ROUND(100.0 * SUM(CASE WHEN c.instructor_name = '' OR c.instructor_name = 'TBA' THEN 1 ELSE 0 END) / COUNT(*), 1) as tba_pct
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%'
    GROUP BY c.term_id ORDER BY c.term_id
""", conn)
fig = px.bar(df_tba, x="term_name", y="tba_pct",
             title="% of Sections With No Assigned Instructor (TBA)",
             labels={"tba_pct": "TBA %", "term_name": "Semester"},
             color="tba_pct", color_continuous_scale="RdYlGn_r")
fig.update_layout(template=TEMPLATE, height=400, xaxis=dict(tickangle=-45, dtick=4))
fig.update_coloraxes(showscale=False)
charts["tba_rate"] = chart_html(fig, "tba_rate")

# ===== 4. Schedule Heatmap =====
df_schedule = pd.read_sql_query("""
    SELECT days, start_time, COUNT(*) as count
    FROM courses WHERE days != '' AND start_time != '' AND start_time != '12:00 am'
    GROUP BY days, start_time
""", conn)
day_map = {"M": "Mon", "T": "Tue", "W": "Wed", "R": "Thu", "U": "Sun", "S": "Sat"}
day_order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Sat"]
rows = []
for _, r in df_schedule.iterrows():
    for ch in r["days"]:
        if ch in day_map:
            rows.append({"day": day_map[ch], "start_time": r["start_time"],
                        "count": r["count"], "time_mins": parse_time_minutes(r["start_time"])})
df_expanded = pd.DataFrame(rows)
df_pivot = df_expanded.pivot_table(index="start_time", columns="day", values="count",
                                    aggfunc="sum", fill_value=0)
df_pivot["time_mins"] = df_pivot.index.map(parse_time_minutes)
df_pivot = df_pivot.sort_values("time_mins").drop("time_mins", axis=1)
df_pivot = df_pivot[[d for d in day_order if d in df_pivot.columns]]
fig = px.imshow(df_pivot.values, x=df_pivot.columns.tolist(), y=df_pivot.index.tolist(),
                color_continuous_scale=[[0, "#ffffff"], [0.15, "#fde8e4"], [0.4, "#e88373"],
                                        [0.7, "#c0392b"], [1.0, "#7b241c"]],
                title="Schedule Heatmap: When Are AUS Courses Held?",
                labels={"x": "Day", "y": "Start Time", "color": "Sections"}, aspect="auto")
fig.update_layout(template=TEMPLATE, height=600)
charts["schedule_heatmap"] = chart_html(fig, "schedule_heatmap")

# ===== 4b. Day patterns =====
df_days = pd.read_sql_query("""
    SELECT days, COUNT(*) as count FROM courses WHERE days != ''
    GROUP BY days ORDER BY count DESC LIMIT 10
""", conn)
day_labels = {"MW": "Mon/Wed", "TRU": "Tue/Thu/Sun", "TR": "Tue/Thu", "M": "Mon",
              "W": "Wed", "T": "Tue", "MTWRU": "Mon-Thu+Sun", "TU": "Tue/Sun",
              "R": "Thu", "U": "Sun", "MWS": "Mon/Wed/Sat"}
df_days["label"] = df_days["days"].map(lambda d: day_labels.get(d, d))
fig = px.pie(df_days, values="count", names="label",
             title="Most Common Day Patterns",
             color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.update_layout(template=TEMPLATE, height=450)
charts["day_patterns"] = chart_html(fig, "day_patterns")

# ===== 4b2. Saturday class decline =====
df_saturday = pd.read_sql_query("""
    SELECT s.term_name, s.term_id,
           SUM(CASE WHEN c.days LIKE '%S%' THEN 1 ELSE 0 END) as saturday_sections,
           COUNT(*) as total_sections
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE (s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%')
    AND c.days IS NOT NULL AND c.days != ''
    GROUP BY c.term_id ORDER BY c.term_id
""", conn)
df_saturday['sat_pct'] = (df_saturday['saturday_sections'] / df_saturday['total_sections'] * 100).round(1)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(
    x=df_saturday['term_name'], y=df_saturday['saturday_sections'],
    name='Saturday Sections', marker_color=AUS_GOLD,
    hovertemplate='%{x}<br>%{y} sections<extra></extra>'), secondary_y=False)
fig.add_trace(go.Scatter(
    x=df_saturday['term_name'], y=df_saturday['sat_pct'],
    name='% of All Sections', mode='lines+markers',
    line=dict(color='#c0392b', width=2), marker=dict(size=4),
    hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'), secondary_y=True)
fig.update_layout(template=TEMPLATE, title="The Disappearance of Saturday Classes",
                  height=450, xaxis=dict(tickangle=-45, dtick=4),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_yaxes(title_text="Saturday Sections", secondary_y=False)
fig.update_yaxes(title_text="% of Total", secondary_y=True)
charts['saturday_decline'] = chart_html(fig, 'saturday_decline')

sat_peak = df_saturday.loc[df_saturday['saturday_sections'].idxmax()]
sat_peak_term = sat_peak['term_name']
sat_peak_count = int(sat_peak['saturday_sections'])
sat_peak_pct = round(sat_peak['sat_pct'], 1)
sat_recent = df_saturday.iloc[-1]
sat_recent_count = int(sat_recent['saturday_sections'])

# ===== 4b3. Day pattern evolution over time =====
df_day_evo = pd.read_sql_query("""
    SELECT s.term_name, s.term_id, c.days, COUNT(*) as cnt
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE c.days IS NOT NULL AND c.days != ''
    AND (s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%')
    GROUP BY s.term_id, c.days ORDER BY s.term_id
""", conn)

def classify_day_pattern(days):
    if days in ('MW',): return 'MW'
    if days in ('TRU',): return 'TRU (Tue/Thu/Sun)'
    if days in ('TR',): return 'TR (Tue/Thu)'
    if 'S' in days: return 'Includes Saturday'
    if days == 'MTWRU': return 'Daily (MTWRU)'
    if len(days) == 1: return 'Single Day'
    return 'Other'

df_day_evo['pattern'] = df_day_evo['days'].apply(classify_day_pattern)
df_day_pattern_agg = df_day_evo.groupby(['term_name', 'term_id', 'pattern'])['cnt'].sum().reset_index()
df_day_total = df_day_pattern_agg.groupby(['term_name', 'term_id'])['cnt'].sum().reset_index().rename(columns={'cnt': 'total'})
df_day_pattern_agg = df_day_pattern_agg.merge(df_day_total)
df_day_pattern_agg['pct'] = (df_day_pattern_agg['cnt'] / df_day_pattern_agg['total'] * 100).round(1)

pattern_colors = {'MW': '#2471a3', 'TRU (Tue/Thu/Sun)': '#c0392b', 'TR (Tue/Thu)': '#27ae60',
                  'Includes Saturday': '#f39c12', 'Daily (MTWRU)': '#8e44ad',
                  'Single Day': '#95a5a6', 'Other': '#bdc3c7'}
pattern_order = ['MW', 'TRU (Tue/Thu/Sun)', 'TR (Tue/Thu)', 'Includes Saturday',
                 'Daily (MTWRU)', 'Single Day', 'Other']

fig = go.Figure()
for pat in pattern_order:
    df_p = df_day_pattern_agg[df_day_pattern_agg['pattern'] == pat].sort_values('term_id')
    if len(df_p) > 0:
        fig.add_trace(go.Scatter(
            x=df_p['term_name'], y=df_p['pct'], name=pat,
            mode='lines', stackgroup='one',
            line=dict(color=pattern_colors.get(pat, '#bdc3c7'), width=0.5),
            hovertemplate='%{x}<br>' + pat + ': %{y:.1f}%<extra></extra>'))
fig.update_layout(template=TEMPLATE, title="Day Pattern Mix Over Time (% of Scheduled Sections)",
                  height=500, xaxis=dict(tickangle=-45, dtick=4),
                  yaxis=dict(title="% of Sections", range=[0, 100]),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts['day_pattern_evolution'] = chart_html(fig, 'day_pattern_evolution')

# ===== 4c. Buildings =====
df_rooms = pd.read_sql_query("""
    SELECT classroom as location, COUNT(*) as count FROM courses
    WHERE classroom != '' AND classroom != 'TBA'
    GROUP BY classroom ORDER BY count DESC
""", conn)
def get_building(loc):
    loc = str(loc)
    if "New Academic" in loc: return "New Academic Bldg 1"
    if "Language" in loc: return "Language Building"
    if "Engineering Building Right" in loc or "EB2" in loc: return "Eng Bldg Right (EB2)"
    if "Engineering Building Left" in loc or "EB1" in loc: return "Eng Bldg Left (EB1)"
    if "Main Building" in loc or "MB" in loc: return "Main Building"
    if "Science" in loc: return "Science Building"
    if "Architecture" in loc or "AB" in loc: return "Architecture Bldg"
    if "Sports" in loc or "Gym" in loc: return "Sports Complex"
    if "Library" in loc: return "Library"
    return "Other"
df_rooms["building"] = df_rooms["location"].apply(get_building)
df_buildings = df_rooms.groupby("building")["count"].sum().sort_values(ascending=False).reset_index()
df_buildings = df_buildings[df_buildings["building"] != "Other"]
fig = px.bar(df_buildings, x="building", y="count",
             title="Course Sections by Building (All Semesters)",
             labels={"count": "Total Sections", "building": ""},
             color="count", color_continuous_scale="Teal")
fig.update_layout(template=TEMPLATE, height=400)
fig.update_coloraxes(showscale=False)
charts["buildings"] = chart_html(fig, "buildings")

# ===== 5. Prerequisites =====
df_deps = pd.read_sql_query("""
    SELECT DISTINCT c.subject as course_subject, c.course_number as course_num,
        d.subject as dep_subject, d.course_number as dep_num, d.dep_type, d.minimum_grade
    FROM course_dependencies d
    JOIN courses c ON c.crn = d.crn AND c.term_id = d.term_id
    WHERE d.dep_type = 'prerequisite'
""", conn)
G = nx.DiGraph()
for _, row in df_deps.iterrows():
    src = f"{row['dep_subject']} {row['dep_num']}"
    dst = f"{row['course_subject']} {row['course_num']}"
    if src != dst:
        G.add_edge(src, dst, grade=row["minimum_grade"])

# Most connected courses
degree_data = []
for node in G.nodes():
    degree_data.append({"course": node, "prerequisites": G.in_degree(node),
                        "is_prereq_for": G.out_degree(node),
                        "total_connections": G.in_degree(node) + G.out_degree(node)})
df_degree = pd.DataFrame(degree_data).sort_values("total_connections", ascending=False)
top25_deg = df_degree.head(25)
fig = go.Figure()
fig.add_trace(go.Bar(y=top25_deg["course"], x=top25_deg["is_prereq_for"],
    name="Is prerequisite for", orientation="h", marker_color="#e74c3c"))
fig.add_trace(go.Bar(y=top25_deg["course"], x=top25_deg["prerequisites"],
    name="Has prerequisites", orientation="h", marker_color="#3498db"))
fig.update_layout(template=TEMPLATE, title="Most Connected Courses in the Prerequisite Graph",
                  xaxis_title="Number of Connections", barmode="stack", height=650,
                  yaxis=dict(autorange="reversed"),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts["prereq_connected"] = chart_html(fig, "prereq_connected")

# ===== 5b. COE network graph =====
dept = "COE"
coe_courses = {n for n in G.nodes() if n.startswith(f"{dept} ")}
coe_extended = set(coe_courses)
for c in coe_courses:
    coe_extended.update(G.predecessors(c))
    coe_extended.update(G.successors(c))
subG = G.subgraph(coe_extended)
pos = nx.spring_layout(subG, k=2, iterations=50, seed=42)
edge_x, edge_y = [], []
for u, v in subG.edges():
    x0, y0 = pos[u]; x1, y1 = pos[v]
    edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
    line=dict(width=0.5, color="#888"), hoverinfo="none")
node_x = [pos[n][0] for n in subG.nodes()]
node_y = [pos[n][1] for n in subG.nodes()]
node_text = list(subG.nodes())
node_color = ["#e74c3c" if n.startswith(f"{dept} ") else "#3498db" for n in subG.nodes()]
node_size = [10 + subG.degree(n) * 2 for n in subG.nodes()]
node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", hoverinfo="text",
    text=node_text, textposition="top center", textfont=dict(size=8),
    marker=dict(color=node_color, size=node_size, line=dict(width=1, color="white")))
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(template=TEMPLATE,
    title=f"{dept} Prerequisite Network (red = {dept}, blue = other depts)",
    showlegend=False, height=700,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
charts["coe_network"] = chart_html(fig, "coe_network")

# ===== 5c. Prereq complexity by dept =====
df_dept_deps = pd.read_sql_query("""
    SELECT c.subject,
           COUNT(DISTINCT d.id) as total_dependencies,
           COUNT(DISTINCT c.subject || c.course_number) as unique_courses,
           ROUND(1.0 * COUNT(DISTINCT d.id) / COUNT(DISTINCT c.subject || c.course_number), 1) as deps_per_course
    FROM course_dependencies d
    JOIN courses c ON c.crn = d.crn AND c.term_id = d.term_id
    WHERE d.dep_type = 'prerequisite'
    GROUP BY c.subject HAVING unique_courses >= 5
    ORDER BY deps_per_course DESC
""", conn)
fig = px.bar(df_dept_deps.head(25), x="subject", y="deps_per_course",
             color="deps_per_course", color_continuous_scale="Reds",
             hover_data=["total_dependencies", "unique_courses"],
             title="Average Prerequisites Per Course by Department",
             labels={"deps_per_course": "Avg Prerequisites/Course", "subject": "Subject"})
fig.update_layout(template=TEMPLATE, height=450)
fig.update_coloraxes(showscale=False)
charts["prereq_complexity"] = chart_html(fig, "prereq_complexity")

# ===== 6. Grade Requirements =====
df_grades = pd.read_sql_query("""
    SELECT minimum_grade, COUNT(*) as count FROM course_dependencies
    WHERE minimum_grade != '' GROUP BY minimum_grade ORDER BY count DESC
""", conn)
grade_order = ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "P"]
df_grades["grade_rank"] = df_grades["minimum_grade"].map({g: i for i, g in enumerate(grade_order)})
df_grades = df_grades.sort_values("grade_rank")
total_deps = df_grades["count"].sum()
df_grades["pct"] = (df_grades["count"] / total_deps * 100).round(1)
# Color grades by severity
grade_colors = {"A": "#922b21", "A-": "#c0392b", "B+": "#d35400", "B": "#e67e22",
                "B-": "#f39c12", "C+": "#2e86c1", "C": "#2471a3", "C-": "#1a5276",
                "D+": "#7f8c8d", "D": "#95a5a6", "D-": "#bdc3c7", "P": "#27ae60"}
colors = [grade_colors.get(g, "#95a5a6") for g in df_grades["minimum_grade"]]
fig = go.Figure(go.Bar(
    y=df_grades["minimum_grade"], x=df_grades["count"], orientation="h",
    text=[f"{row['count']:,} ({row['pct']}%)" for _, row in df_grades.iterrows()],
    textposition="outside", marker_color=colors,
    hovertemplate="%{y}: %{x:,} dependencies<extra></extra>"))
fig.update_layout(template=TEMPLATE, height=500,
                  title="Minimum Grade Requirements Across All Prerequisites",
                  xaxis_title="Number of Dependencies", yaxis_title="",
                  yaxis=dict(categoryorder="array",
                             categoryarray=list(reversed(grade_order))),
                  margin=dict(r=160))
charts["grades"] = chart_html(fig, "grades")

# ===== 6b. Grade strictness by dept =====
df_strict = pd.read_sql_query("""
    SELECT c.subject,
           SUM(CASE WHEN d.minimum_grade IN ('A','A-') THEN 1 ELSE 0 END) as grade_A,
           SUM(CASE WHEN d.minimum_grade IN ('B+','B','B-') THEN 1 ELSE 0 END) as grade_B,
           SUM(CASE WHEN d.minimum_grade IN ('C+','C') THEN 1 ELSE 0 END) as grade_C,
           SUM(CASE WHEN d.minimum_grade = 'C-' THEN 1 ELSE 0 END) as grade_C_minus,
           SUM(CASE WHEN d.minimum_grade IN ('D+','D','D-') THEN 1 ELSE 0 END) as grade_D,
           COUNT(*) as total
    FROM course_dependencies d
    JOIN courses c ON c.crn = d.crn AND c.term_id = d.term_id
    WHERE d.minimum_grade != ''
    GROUP BY c.subject HAVING total >= 50
    ORDER BY 1.0 * (grade_A + grade_B) / total DESC
""", conn)
# Convert to percentages for meaningful comparison
for col in ["grade_A", "grade_B", "grade_C", "grade_C_minus", "grade_D"]:
    df_strict[f"{col}_pct"] = (df_strict[col] / df_strict["total"] * 100).round(1)
fig = go.Figure()
for col, color, label in [("grade_A_pct", "#c0392b", "A/A-"), ("grade_B_pct", "#e67e22", "B range"),
    ("grade_C_pct", "#2471a3", "C/C+"), ("grade_C_minus_pct", "#27ae60", "C-"), ("grade_D_pct", "#95a5a6", "D range")]:
    fig.add_trace(go.Bar(x=df_strict["subject"], y=df_strict[col], name=label, marker_color=color,
                         hovertemplate="%{x}: %{y:.1f}%<extra>" + label + "</extra>"))
fig.update_layout(template=TEMPLATE, title="Grade Requirement Distribution by Department (%)",
                  barmode="stack", height=500, xaxis_title="Department", yaxis_title="% of Prerequisites",
                  yaxis=dict(range=[0, 100]),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts["grade_strictness"] = chart_html(fig, "grade_strictness")

# ===== 7. Credit hours =====
df_catalog = pd.read_sql_query("""
    SELECT subject, course_number, description, credit_hours,
           lecture_hours, lab_hours, department
    FROM catalog WHERE credit_hours > 0 AND credit_hours < 30
""", conn)
fig = px.histogram(df_catalog, x="credit_hours", nbins=20,
                   title="Credit Hours Distribution Across All Courses",
                   labels={"credit_hours": "Credit Hours", "count": "Courses"},
                   color_discrete_sequence=[AUS_GOLD])
fig.update_layout(template=TEMPLATE, height=400)
charts["credit_hours"] = chart_html(fig, "credit_hours")

# ===== 7b. Lecture vs Lab hours =====
df_dept_hours = df_catalog.groupby("department").agg(
    {"lecture_hours": "mean", "lab_hours": "mean", "credit_hours": ["mean", "count"]}).reset_index()
df_dept_hours.columns = ["department", "avg_lecture", "avg_lab", "avg_credits", "course_count"]
df_dept_hours = df_dept_hours[df_dept_hours["course_count"] >= 10].sort_values("avg_lecture", ascending=True)
df_dept_hours["dept_short"] = df_dept_hours["department"].str.replace(" Department", "").str.replace(" (n.a.)", "", regex=False)
fig = go.Figure()
fig.add_trace(go.Bar(y=df_dept_hours["dept_short"], x=df_dept_hours["avg_lecture"],
    name="Avg Lecture Hrs", orientation="h", marker_color="#2471a3",
    text=df_dept_hours["avg_lecture"].round(1), textposition="outside"))
fig.add_trace(go.Bar(y=df_dept_hours["dept_short"], x=df_dept_hours["avg_lab"],
    name="Avg Lab Hrs", orientation="h", marker_color="#c0392b",
    text=df_dept_hours["avg_lab"].round(1), textposition="outside"))
fig.update_layout(template=TEMPLATE, title="Average Lecture vs Lab Hours by Department",
                  barmode="group", height=600, xaxis_title="Hours",
                  margin=dict(r=80),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts["lecture_lab"] = chart_html(fig, "lecture_lab")

# ===== 8. Section closure (NOT enrollment counts — see note) =====
# Banner exposes only a binary "open seats? Y/N" flag at crawl time, never a
# seat count or enrolment total. For a *completed* term this flag effectively
# records whether the section closed (ran out of open seats); for the term
# that is still in open registration it is a live, not-yet-final snapshot, so
# we exclude that most-recent regular term from the closure trend.
latest_regular_term = pd.read_sql_query("""
    SELECT MAX(s.term_id) AS t FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%'
""", conn).iloc[0]["t"]

df_seats = pd.read_sql_query("""
    SELECT s.term_name, s.term_id,
           SUM(CASE WHEN c.seats_available = 1 THEN 1 ELSE 0 END) as available,
           SUM(CASE WHEN c.seats_available = 0 THEN 1 ELSE 0 END) as full_sections,
           COUNT(*) as total,
           ROUND(100.0 * SUM(CASE WHEN c.seats_available = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as full_pct
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE (s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%')
      AND c.term_id < ?
    GROUP BY c.term_id ORDER BY c.term_id
""", conn, params=[latest_regular_term])
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=df_seats["term_name"], y=df_seats["available"],
    name="Had open seats", marker_color="#2ecc71"), secondary_y=False)
fig.add_trace(go.Bar(x=df_seats["term_name"], y=df_seats["full_sections"],
    name="Closed (no open seats)", marker_color="#e74c3c"), secondary_y=False)
fig.add_trace(go.Scatter(x=df_seats["term_name"], y=df_seats["full_pct"],
    name="% closed", mode="lines+markers", line=dict(color="#7b241c", width=2),
    marker=dict(size=4)), secondary_y=True)
fig.update_layout(template=TEMPLATE, title="Section Closure Per Semester (Completed Terms)",
                  barmode="stack", height=450, xaxis=dict(tickangle=-45, dtick=4),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_yaxes(title_text="Sections", secondary_y=False)
fig.update_yaxes(title_text="% Closed", secondary_y=True, range=[0, 100])
charts["enrollment"] = chart_html(fig, "enrollment")

closure_early = float(df_seats["full_pct"].iloc[:6].mean())
closure_recent = float(df_seats["full_pct"].iloc[-6:].mean())

# ===== 8b. Closure rate by subject (completed terms only) =====
df_subj_full = pd.read_sql_query("""
    SELECT subject,
           SUM(CASE WHEN seats_available = 0 THEN 1 ELSE 0 END) as full_count,
           COUNT(*) as total,
           ROUND(100.0 * SUM(CASE WHEN seats_available = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as full_pct
    FROM courses WHERE term_id < ?
    GROUP BY subject HAVING total >= 50 ORDER BY full_pct DESC
""", conn, params=[latest_regular_term])
fig = px.bar(df_subj_full.head(25), x="subject", y="full_pct",
             color="full_pct", color_continuous_scale="RdYlGn_r",
             hover_data=["full_count", "total"],
             title="Section Closure Rate by Subject (% of Sections That Filled)",
             labels={"full_pct": "% Closed", "subject": "Subject"})
fig.update_layout(template=TEMPLATE, height=450)
fig.update_coloraxes(showscale=False)
charts["fill_rate"] = chart_html(fig, "fill_rate")

# ===== 9. Fees =====
df_fees_raw = pd.read_sql_query("""
    SELECT sd.crn, sd.term_id, sd.fees, c.subject, c.title, s.term_name
    FROM section_details sd
    JOIN courses c ON c.crn = sd.crn AND c.term_id = sd.term_id
    JOIN semesters s ON s.term_id = sd.term_id
    WHERE sd.fees != '' AND sd.fees != '[]'
""", conn)
fee_records = []
for _, row in df_fees_raw.iterrows():
    try:
        fees = json.loads(row["fees"])
        for fee in fees:
            fee_records.append({"subject": row["subject"], "title": row["title"],
                "term_name": row["term_name"], "term_id": row["term_id"],
                "fee_type": fee.get("description", ""), "amount": float(fee.get("amount", 0))})
    except (json.JSONDecodeError, TypeError):
        pass
df_fees = pd.DataFrame(fee_records) if fee_records else pd.DataFrame()
if len(df_fees) > 0:
    # Shorten long fee type names for readability
    df_fees["fee_short"] = df_fees["fee_type"].str.replace("Technology Fee - ", "Tech Fee: ")
    fee_colors = ["#c0392b", "#2471a3", "#27ae60", "#8e44ad", "#d68910",
                  "#1abc9c", "#e74c3c", "#2c3e50"]
    # Focus on the most common fee types and filter extreme outliers (tuition etc.)
    top_fee_types = df_fees["fee_short"].value_counts().head(10).index.tolist()
    df_fees_top = df_fees[df_fees["fee_short"].isin(top_fee_types) & (df_fees["amount"] < 5000)]
    fig = px.box(df_fees_top, y="fee_short", x="amount", orientation="h",
                 title="Fee Amount Distribution (Top 10 Fee Types, under 5,000 AED)",
                 labels={"amount": "Amount (AED)", "fee_short": ""},
                 color="fee_short", color_discrete_sequence=fee_colors)
    fig.update_layout(template=TEMPLATE, height=450, showlegend=False,
                      margin=dict(l=200))
    charts["fees"] = chart_html(fig, "fees")

    # For fee trend, use consistent colors per fee type (no mid-line color change)
    df_fee_trend = df_fees.groupby(["term_id", "term_name", "fee_short"]).agg(
        avg_amount=("amount", "mean"), count=("amount", "count")).reset_index().sort_values("term_id")
    top_fees = df_fees["fee_short"].value_counts().head(5).index.tolist()
    df_fee_trend_top = df_fee_trend[df_fee_trend["fee_short"].isin(top_fees)]
    fig = go.Figure()
    for i, fee_type in enumerate(top_fees):
        df_ft = df_fee_trend_top[df_fee_trend_top["fee_short"] == fee_type].sort_values("term_id")
        fig.add_trace(go.Scatter(
            x=df_ft["term_name"], y=df_ft["avg_amount"],
            mode="lines+markers", name=fee_type,
            line=dict(color=fee_colors[i % len(fee_colors)], width=2),
            marker=dict(size=5)))
    fig.update_layout(template=TEMPLATE, height=450,
                      title="Average Fee Amount Over Time",
                      xaxis_title="Semester", yaxis_title="Amount (AED)",
                      xaxis=dict(tickangle=-45, dtick=4),
                      legend=dict(font=dict(size=10)))
    charts["fee_trend"] = chart_html(fig, "fee_trend")

# ===== 10. Cross-dept dependencies =====
df_cross = pd.read_sql_query("""
    SELECT DISTINCT c.subject as course_dept, d.subject as prereq_dept,
           COUNT(DISTINCT c.subject || c.course_number) as course_count
    FROM course_dependencies d
    JOIN courses c ON c.crn = d.crn AND c.term_id = d.term_id
    WHERE d.dep_type = 'prerequisite' AND c.subject != d.subject
    GROUP BY c.subject, d.subject HAVING course_count >= 3 ORDER BY course_count DESC
""", conn)
df_matrix = df_cross.pivot_table(index="course_dept", columns="prereq_dept",
                                  values="course_count", fill_value=0)
common_depts = df_matrix.sum(axis=1).sort_values(ascending=False).head(20).index.tolist()
common_prereqs = df_matrix.sum(axis=0).sort_values(ascending=False).head(20).index.tolist()
all_relevant = sorted(set(common_depts + common_prereqs))
df_mf = df_matrix.reindex(index=all_relevant, columns=all_relevant, fill_value=0)
df_mf = df_mf.loc[(df_mf.sum(axis=1) > 0), (df_mf.sum(axis=0) > 0)]
fig = px.imshow(df_mf, color_continuous_scale="Blues",
                title="Cross-Department Prerequisite Dependencies",
                labels={"x": "Prerequisite Dept", "y": "Course Dept", "color": "Courses"}, aspect="auto")
fig.update_layout(template=TEMPLATE, height=600)
charts["cross_dept"] = chart_html(fig, "cross_dept")

# ===== 11. Lab vs Lecture =====
df_lab = pd.read_sql_query("""
    SELECT s.term_name, s.term_id,
           SUM(CASE WHEN c.is_lab = 1 THEN 1 ELSE 0 END) as lab_sections,
           SUM(CASE WHEN c.is_lab = 0 THEN 1 ELSE 0 END) as lecture_sections,
           COUNT(*) as total
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%'
    GROUP BY c.term_id ORDER BY c.term_id
""", conn)
df_lab["lab_pct"] = df_lab["lab_sections"] / df_lab["total"] * 100
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=df_lab["term_name"], y=df_lab["lecture_sections"],
    name="Lectures", marker_color="#3498db"), secondary_y=False)
fig.add_trace(go.Bar(x=df_lab["term_name"], y=df_lab["lab_sections"],
    name="Labs", marker_color="#e74c3c"), secondary_y=False)
fig.add_trace(go.Scatter(x=df_lab["term_name"], y=df_lab["lab_pct"],
    name="Lab %", mode="lines+markers", line=dict(color="#2ecc71", width=2),
    marker=dict(size=4)), secondary_y=True)
fig.update_layout(template=TEMPLATE, title="Lecture vs Lab Sections Over Time",
                  barmode="stack", height=450, xaxis=dict(tickangle=-45, dtick=4),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_yaxes(title_text="Sections", secondary_y=False)
fig.update_yaxes(title_text="Lab %", secondary_y=True)
charts["lab_lecture"] = chart_html(fig, "lab_lecture")

# Longest prereq chains — use DAG version to avoid cycle issues
longest_chains = []
try:
    # Break cycles by working on a DAG copy
    dag = G.copy()
    while not nx.is_directed_acyclic_graph(dag):
        cycle = nx.find_cycle(dag)
        dag.remove_edge(*cycle[0][:2])
    longest = {}
    for node in nx.topological_sort(dag):
        preds = list(dag.predecessors(node))
        if not preds:
            longest[node] = [node]
        else:
            best = max((longest.get(p, [p]) for p in preds), key=len)
            longest[node] = best + [node]
    sorted_paths = sorted(longest.items(), key=lambda x: len(x[1]), reverse=True)
    longest_chains = sorted_paths[:12]
except Exception:
    pass

# ===== NEW: Academic Levels Over Time =====
df_levels_raw = pd.read_sql_query("""
    SELECT c.levels, s.term_name, s.term_id, COUNT(*) as sections
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE c.levels != '' AND (s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%')
    GROUP BY c.levels, s.term_id ORDER BY s.term_id
""", conn)

def primary_level(lev):
    lev = str(lev).lower()
    if 'doctorate' in lev: return 'Doctorate'
    if 'graduate' in lev and 'under' not in lev: return 'Graduate'
    if 'achievement' in lev: return 'Achievement Academy'
    if 'intensive english' in lev: return 'Intensive English'
    if 'post bachelor' in lev: return 'Post Bachelor'
    return 'Undergraduate'

df_levels_raw['primary_level'] = df_levels_raw['levels'].apply(primary_level)
df_levels_agg = df_levels_raw.groupby(['term_name', 'term_id', 'primary_level'])['sections'].sum().reset_index()

level_order = ['Undergraduate', 'Graduate', 'Post Bachelor', 'Doctorate',
               'Achievement Academy', 'Intensive English']
level_colors_map = {'Undergraduate': '#2471a3', 'Graduate': '#c0392b', 'Post Bachelor': '#8e44ad',
                    'Doctorate': '#27ae60', 'Achievement Academy': '#d68910', 'Intensive English': '#1abc9c'}

fig = go.Figure()
for level in level_order:
    df_l = df_levels_agg[df_levels_agg['primary_level'] == level].sort_values('term_id')
    if len(df_l) > 0:
        fig.add_trace(go.Scatter(x=df_l['term_name'], y=df_l['sections'],
            name=level, mode='lines', stackgroup='one',
            line=dict(color=level_colors_map.get(level, '#95a5a6'), width=0.5)))
fig.update_layout(template=TEMPLATE, title="Course Sections by Academic Level Over Time",
                  height=500, xaxis=dict(tickangle=-45, dtick=4),
                  yaxis_title="Sections",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts['levels_over_time'] = chart_html(fig, 'levels_over_time')

# Level distribution (aggregate)
df_level_totals = df_levels_raw.groupby('primary_level')['sections'].sum().reset_index()
df_level_totals = df_level_totals.sort_values('sections', ascending=True)
total_level_sections = df_level_totals['sections'].sum()
df_level_totals['pct'] = (df_level_totals['sections'] / total_level_sections * 100).round(1)
undergrad_pct = df_level_totals.loc[df_level_totals['primary_level'] == 'Undergraduate', 'pct'].iloc[0]
grad_total = df_level_totals.loc[df_level_totals['primary_level'] == 'Graduate', 'sections'].iloc[0] if 'Graduate' in df_level_totals['primary_level'].values else 0

fig = go.Figure(go.Bar(
    y=df_level_totals['primary_level'], x=df_level_totals['sections'], orientation='h',
    text=[f"{row['sections']:,} ({row['pct']}%)" for _, row in df_level_totals.iterrows()],
    textposition='outside',
    marker_color=[level_colors_map.get(l, '#95a5a6') for l in df_level_totals['primary_level']]))
fig.update_layout(template=TEMPLATE, title="Total Sections by Academic Level (All Semesters)",
                  height=400, xaxis_title="Total Sections", margin=dict(r=140))
charts['levels_dist'] = chart_html(fig, 'levels_dist')

# Level mix by subject (percentage stacked bar)
df_level_subj = pd.read_sql_query("""
    SELECT c.subject, c.levels, COUNT(*) as sections
    FROM courses c WHERE c.levels != ''
    GROUP BY c.subject, c.levels
""", conn)
df_level_subj['primary_level'] = df_level_subj['levels'].apply(primary_level)
df_level_subj_agg = df_level_subj.groupby(['subject', 'primary_level'])['sections'].sum().reset_index()
top20_for_levels = df_level_subj_agg.groupby('subject')['sections'].sum().nlargest(20).index.tolist()
df_lv_top = df_level_subj_agg[df_level_subj_agg['subject'].isin(top20_for_levels)]
df_lv_pivot = df_lv_top.pivot_table(index='subject', columns='primary_level', values='sections', fill_value=0)
df_lv_pct = df_lv_pivot.div(df_lv_pivot.sum(axis=1), axis=0) * 100
sort_col = 'Graduate' if 'Graduate' in df_lv_pct.columns else df_lv_pct.columns[0]
df_lv_pct = df_lv_pct.sort_values(sort_col, ascending=True)

fig = go.Figure()
for level in level_order:
    if level in df_lv_pct.columns:
        fig.add_trace(go.Bar(y=df_lv_pct.index, x=df_lv_pct[level],
            name=level, orientation='h', marker_color=level_colors_map.get(level, '#95a5a6'),
            hovertemplate="%{y}: %{x:.1f}%<extra>" + level + "</extra>"))
fig.update_layout(template=TEMPLATE, title="Academic Level Mix by Subject (Top 20)",
                  barmode='stack', height=600, xaxis_title="% of Sections",
                  xaxis=dict(range=[0, 100]),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts['levels_by_subject'] = chart_html(fig, 'levels_by_subject')

# ===== NEW: Instructor Diversity Scatter =====
fig = px.scatter(df_instructors.head(200), x='total_sections', y='subjects_taught',
                 size='semesters_active', hover_name='instructor_name',
                 color='semesters_active', color_continuous_scale=['#d4e6f1', '#1a5276'],
                 labels={'total_sections': 'Total Sections Taught', 'subjects_taught': 'Distinct Subjects',
                         'semesters_active': 'Semesters Active'},
                 title="Instructor Teaching Diversity (Top 200 by Sections)")
fig.update_layout(template=TEMPLATE, height=500,
                  coloraxis_colorbar=dict(title="Semesters", thickness=15, len=0.5))
charts['instructor_diversity'] = chart_html(fig, 'instructor_diversity')

# Average sections per instructor over time
df_inst_workload = pd.read_sql_query("""
    SELECT s.term_name, s.term_id,
           COUNT(*) * 1.0 / COUNT(DISTINCT c.instructor_name) as avg_sections
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE c.instructor_name != '' AND c.instructor_name != 'TBA'
    AND (s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%')
    GROUP BY c.term_id ORDER BY c.term_id
""", conn)
fig = px.line(df_inst_workload, x='term_name', y='avg_sections',
              title="Average Sections Per Instructor Per Semester",
              labels={'avg_sections': 'Avg Sections/Instructor', 'term_name': 'Semester'},
              color_discrete_sequence=[AUS_GOLD])
fig.update_traces(line=dict(width=2.5))
fig.update_layout(template=TEMPLATE, height=400, xaxis=dict(tickangle=-45, dtick=4))
charts['instructor_workload'] = chart_html(fig, 'instructor_workload')

# ===== NEW: Instructor Recruitment & Departure =====
# First and last semester for each instructor
df_inst_career = pd.read_sql_query("""
    SELECT instructor_name,
           MIN(c.term_id) as first_term_id, MAX(c.term_id) as last_term_id
    FROM courses c
    WHERE c.instructor_name != '' AND c.instructor_name != 'TBA'
    GROUP BY instructor_name
""", conn)

# Get all major semesters in order
df_sem_order = pd.read_sql_query("""
    SELECT term_id, term_name FROM semesters
    WHERE term_name LIKE 'Fall%' OR term_name LIKE 'Spring%'
    ORDER BY term_id
""", conn)
sem_list = df_sem_order['term_id'].tolist()
sem_names = dict(zip(df_sem_order['term_id'], df_sem_order['term_name']))

# Count new hires and departures per semester
new_hires = df_inst_career.groupby('first_term_id').size().reindex(sem_list, fill_value=0)
# Departure = last_term_id is this semester (but NOT if it's the most recent semester)
last_two_semesters = sem_list[-2:]  # Exclude recent — they haven't "departed" yet
departures = df_inst_career[~df_inst_career['last_term_id'].isin(last_two_semesters)].groupby('last_term_id').size().reindex(sem_list, fill_value=0)

df_turnover = pd.DataFrame({
    'term_id': sem_list,
    'term_name': [sem_names[t] for t in sem_list],
    'new_hires': new_hires.values,
    'departures': departures.values
})

fig = go.Figure()
fig.add_trace(go.Bar(x=df_turnover['term_name'], y=df_turnover['new_hires'],
    name='New Instructors', marker_color='#27ae60',
    hovertemplate='%{x}<br>%{y} new instructors<extra></extra>'))
fig.add_trace(go.Bar(x=df_turnover['term_name'], y=-df_turnover['departures'],
    name='Departures', marker_color='#c0392b',
    hovertemplate='%{x}<br>%{customdata} departures<extra></extra>',
    customdata=df_turnover['departures']))
fig.update_layout(template=TEMPLATE, title="Instructor Recruitment & Departures Per Semester",
                  height=480, barmode='relative',
                  xaxis=dict(tickangle=-45, dtick=4),
                  yaxis_title="Instructors (positive = hired, negative = departed)",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts['instructor_turnover'] = chart_html(fig, 'instructor_turnover')

total_ever = len(df_inst_career)
still_active = len(df_inst_career[df_inst_career['last_term_id'].isin(last_two_semesters)])

# ===== NEW: Instructor Retention Curve =====
# For each hiring cohort (year), what % are still teaching N years later?
df_inst_career['first_year'] = df_inst_career['first_term_id'].str[:4].astype(int)
df_inst_career['last_year'] = df_inst_career['last_term_id'].str[:4].astype(int)
df_inst_career['tenure_years'] = df_inst_career['last_year'] - df_inst_career['first_year']

# Group into cohorts by 5-year bands
cohort_bins = [(2005, 2009), (2010, 2014), (2015, 2019)]
cohort_colors = {'2005-2009': '#2471a3', '2010-2014': '#27ae60', '2015-2019': '#8e44ad'}
max_years = 15

fig = go.Figure()
for start, end in cohort_bins:
    label = f'{start}-{end}'
    cohort = df_inst_career[(df_inst_career['first_year'] >= start) & (df_inst_career['first_year'] <= end)]
    total = len(cohort)
    if total < 10:
        continue
    survival = []
    for y in range(0, max_years + 1):
        remaining = len(cohort[cohort['tenure_years'] >= y])
        survival.append(round(remaining / total * 100, 1))
    fig.add_trace(go.Scatter(
        x=list(range(max_years + 1)), y=survival,
        name=f'{label} ({total} instructors)',
        mode='lines+markers', marker=dict(size=5),
        line=dict(color=cohort_colors.get(label, '#95a5a6'), width=2.5),
        hovertemplate='Year %{x}<br>%{y:.1f}% remaining<extra>' + label + '</extra>'))

fig.update_layout(template=TEMPLATE, title="Instructor Retention Curves by Hiring Cohort",
                  height=450,
                  xaxis=dict(title="Years After First Semester", dtick=2),
                  yaxis=dict(title="% Still Teaching", range=[0, 105]),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts['instructor_retention'] = chart_html(fig, 'instructor_retention')

# ===== NEW: Course Ownership =====
# For each course, count distinct instructors vs semesters offered
df_ownership = pd.read_sql_query("""
    SELECT subject, course_number, title,
           COUNT(DISTINCT instructor_name) as num_instructors,
           COUNT(DISTINCT term_id) as num_terms,
           COUNT(*) as total_sections
    FROM courses
    WHERE instructor_name != '' AND instructor_name != 'TBA'
    GROUP BY subject, course_number
    HAVING num_terms >= 5
""", conn)
df_ownership['course'] = df_ownership['subject'] + ' ' + df_ownership['course_number']
df_ownership['instructor_per_term'] = (df_ownership['num_instructors'] / df_ownership['num_terms']).round(2)

# Scatter: terms offered vs distinct instructors (log scale helps with density)
fig = px.scatter(df_ownership, x='num_terms', y='num_instructors',
                 size='total_sections', size_max=20,
                 hover_data=['course', 'title', 'total_sections'],
                 color='instructor_per_term',
                 color_continuous_scale=[[0, '#27ae60'], [0.5, '#f1c40f'], [1.0, '#c0392b']],
                 labels={'num_terms': 'Semesters Offered', 'num_instructors': 'Distinct Instructors',
                         'instructor_per_term': 'Instructors<br>per Term', 'total_sections': 'Total Sections'},
                 title="Course Ownership: Instructor Continuity vs. Turnover")
fig.update_layout(template=TEMPLATE, height=520)
charts['course_ownership'] = chart_html(fig, 'course_ownership')

# Top "single-owner" courses (1 instructor, many terms)
df_single_owner = pd.read_sql_query("""
    SELECT c.subject || ' ' || c.course_number as course, c.title,
           c.instructor_name, COUNT(DISTINCT c.term_id) as terms_taught
    FROM courses c
    WHERE c.instructor_name != '' AND c.instructor_name != 'TBA'
    GROUP BY c.subject, c.course_number, c.instructor_name
    HAVING terms_taught >= 15
    ORDER BY terms_taught DESC LIMIT 20
""", conn)

fig = go.Figure(go.Bar(
    y=[f"{r['course']} — {r['instructor_name']}" for _, r in df_single_owner.iterrows()],
    x=df_single_owner['terms_taught'], orientation='h',
    marker_color=AUS_GOLD,
    hovertemplate='%{y}<br>%{x} semesters<extra></extra>'))
fig.update_layout(template=TEMPLATE,
                  title="Longest Instructor-Course Pairings (Same Person, Same Course)",
                  height=max(450, len(df_single_owner) * 24),
                  xaxis_title="Semesters Taught",
                  yaxis=dict(autorange='reversed'))
charts['course_ownership_top'] = chart_html(fig, 'course_ownership_top')

most_owned = df_single_owner.iloc[0]
most_owned_course = most_owned['course']
most_owned_instructor = most_owned['instructor_name']
most_owned_terms = int(most_owned['terms_taught'])

# ===== NEW: Teaching Modality Over Time =====
df_modality = pd.read_sql_query("""
    SELECT s.term_name, s.term_id, c.instructional_method, COUNT(*) as sections
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE c.instructional_method != ''
    AND (s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%')
    GROUP BY s.term_id, c.instructional_method ORDER BY s.term_id
""", conn)

def classify_modality(m):
    m = str(m).strip()
    if m == 'Traditional': return 'Traditional'
    if 'Non-traditional' in m or 'Non Traditional' in m: return 'Non-Traditional'
    if 'Blended' in m: return 'Blended'
    if 'On-Line' in m or 'Online' in m: return 'Online'
    return 'Other'

df_modality['mode'] = df_modality['instructional_method'].apply(classify_modality)
df_mod_agg = df_modality.groupby(['term_name', 'term_id', 'mode'])['sections'].sum().reset_index()
modality_colors = {'Traditional': '#2471a3', 'Non-Traditional': '#c0392b', 'Blended': '#8e44ad',
                   'Online': '#27ae60', 'Other': '#95a5a6'}

# Calculate percentages for stacked bar
df_mod_total = df_mod_agg.groupby(['term_name', 'term_id'])['sections'].sum().reset_index().rename(columns={'sections': 'total'})
df_mod_merged = df_mod_agg.merge(df_mod_total)
df_mod_merged['pct'] = (df_mod_merged['sections'] / df_mod_merged['total'] * 100).round(1)

fig = go.Figure()
for mode in ['Traditional', 'Non-Traditional', 'Blended', 'Online', 'Other']:
    df_m = df_mod_merged[df_mod_merged['mode'] == mode].sort_values('term_id')
    if len(df_m) > 0:
        fig.add_trace(go.Bar(x=df_m['term_name'], y=df_m['pct'], name=mode,
            marker_color=modality_colors.get(mode, '#95a5a6'),
            hovertemplate="%{x}: %{y:.1f}%<extra>" + mode + "</extra>"))
fig.update_layout(template=TEMPLATE, title="Teaching Modality Distribution Over Time (%)",
                  barmode='stack', height=500, xaxis=dict(tickangle=-45, dtick=4),
                  yaxis_title="% of Sections", yaxis=dict(range=[0, 100]),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts['modality_over_time'] = chart_html(fig, 'modality_over_time')

# Non-traditional rate by subject
df_nontrad_subj = pd.read_sql_query("""
    SELECT c.subject,
           SUM(CASE WHEN c.instructional_method != 'Traditional' AND c.instructional_method != '' THEN 1 ELSE 0 END) as nontrad,
           COUNT(*) as total
    FROM courses c WHERE c.instructional_method != ''
    GROUP BY c.subject HAVING total >= 50
""", conn)
df_nontrad_subj['nontrad_pct'] = (df_nontrad_subj['nontrad'] / df_nontrad_subj['total'] * 100).round(1)
df_nontrad_subj = df_nontrad_subj.sort_values('nontrad_pct', ascending=False).head(20)

fig = px.bar(df_nontrad_subj, x='subject', y='nontrad_pct',
             color='nontrad_pct', color_continuous_scale='RdYlBu_r',
             hover_data=['nontrad', 'total'],
             title="Non-Traditional Teaching Rate by Subject (Top 20)",
             labels={'nontrad_pct': '% Non-Traditional', 'subject': 'Subject'})
fig.update_layout(template=TEMPLATE, height=450)
fig.update_coloraxes(showscale=False)
charts['modality_by_subject'] = chart_html(fig, 'modality_by_subject')

# Modality stats
total_nontrad = df_modality[df_modality['mode'] != 'Traditional']['sections'].sum()
total_modality = df_modality['sections'].sum()
nontrad_overall_pct = round(total_nontrad / total_modality * 100, 1) if total_modality > 0 else 0

# ===== COVID-19 Impact (data-driven) =====
# Query all major semesters with multiple metrics
df_covid = pd.read_sql_query("""
    SELECT s.term_name, s.term_id, COUNT(*) as sections,
           COUNT(DISTINCT c.subject || c.course_number) as unique_courses,
           COUNT(DISTINCT c.instructor_name) as instructors,
           SUM(CASE WHEN c.classroom IS NULL OR c.classroom = '' OR c.classroom = 'TBA' THEN 1 ELSE 0 END) as no_classroom,
           ROUND(100.0 * SUM(CASE WHEN c.classroom IS NULL OR c.classroom = '' OR c.classroom = 'TBA' THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_no_classroom
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%'
    GROUP BY c.term_id ORDER BY c.term_id
""", conn)

# --- Chart 1: Sections timeline with COVID band ---
# Simple line chart with shaded COVID region — no colored era bars
covid_start_idx = df_covid[df_covid['term_name'] == 'Spring 2020'].index
covid_end_idx = df_covid[df_covid['term_name'] == 'Spring 2021'].index

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_covid['term_name'], y=df_covid['sections'],
    mode='lines+markers', name='Total Sections',
    line=dict(color='#2471a3', width=2.5), marker=dict(size=5)))
fig.add_trace(go.Scatter(
    x=df_covid['term_name'], y=df_covid['unique_courses'],
    mode='lines+markers', name='Unique Courses',
    line=dict(color='#c0392b', width=2, dash='dash'), marker=dict(size=4)))
# Add COVID shading
if len(covid_start_idx) > 0 and len(covid_end_idx) > 0:
    fig.add_vrect(x0='Spring 2020', x1='Spring 2021',
                  fillcolor='rgba(192,57,43,0.12)', line_width=0,
                  annotation_text='COVID', annotation_position='top left',
                  annotation=dict(font_size=11, font_color='#c0392b'))
fig.update_layout(template=TEMPLATE,
                  title="Course Sections & Unique Courses Per Semester",
                  height=500, xaxis=dict(tickangle=-45, dtick=4),
                  yaxis_title="Count",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts['covid_sections'] = chart_html(fig, 'covid_sections')

# Compute stats for narrative
f19_sections = int(df_covid[df_covid['term_name'] == 'Fall 2019']['sections'].iloc[0])
f20_sections = int(df_covid[df_covid['term_name'] == 'Fall 2020']['sections'].iloc[0])
latest_sections = int(df_covid['sections'].iloc[-1])
latest_term = df_covid['term_name'].iloc[-1]
covid_section_change = round((f20_sections / f19_sections - 1) * 100, 1)
growth_since = round((latest_sections / f19_sections - 1) * 100, 1)
f19_courses = int(df_covid[df_covid['term_name'] == 'Fall 2019']['unique_courses'].iloc[0])
s21_courses = int(df_covid[df_covid['term_name'] == 'Spring 2021']['unique_courses'].iloc[0])
course_variety_drop = round((s21_courses / f19_courses - 1) * 100, 1)

# --- Chart 2: Instructors + Sections-per-course (variety contraction) ---
df_covid['sections_per_course'] = (df_covid['sections'] / df_covid['unique_courses']).round(2)
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(
    x=df_covid['term_name'], y=df_covid['instructors'],
    mode='lines+markers', name='Unique Instructors',
    line=dict(color='#8e44ad', width=2.5), marker=dict(size=5)),
    secondary_y=False)
fig.add_trace(go.Scatter(
    x=df_covid['term_name'], y=df_covid['unique_courses'],
    mode='lines+markers', name='Unique Courses',
    line=dict(color='#27ae60', width=2, dash='dash'), marker=dict(size=4)),
    secondary_y=True)
fig.add_vrect(x0='Spring 2020', x1='Spring 2021',
              fillcolor='rgba(192,57,43,0.12)', line_width=0)
fig.update_layout(template=TEMPLATE,
                  title="Unique Instructors & Course Variety Over Time",
                  height=480, xaxis=dict(tickangle=-45, dtick=4),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_yaxes(title_text="Unique Instructors", secondary_y=False)
fig.update_yaxes(title_text="Unique Courses", secondary_y=True)
charts['covid_variety'] = chart_html(fig, 'covid_variety')

f19_inst = int(df_covid[df_covid['term_name'] == 'Fall 2019']['instructors'].iloc[0])
f20_inst = int(df_covid[df_covid['term_name'] == 'Fall 2020']['instructors'].iloc[0])
inst_change = round((f20_inst / f19_inst - 1) * 100, 1)

# --- Chart 3: Subject-level heatmap Fall 2018 → Fall 2022, indexed to Fall 2019 = 100 ---
compare_terms = ['Fall 2018', 'Fall 2019', 'Fall 2020', 'Fall 2021', 'Fall 2022']
df_subj_covid = pd.read_sql_query(f"""
    SELECT c.subject, s.term_name, COUNT(*) as sections
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE s.term_name IN ({','.join('?' for _ in compare_terms)})
    GROUP BY c.subject, s.term_name
""", conn, params=compare_terms)

pivot = df_subj_covid.pivot_table(index='subject', columns='term_name', values='sections', fill_value=0)
pivot = pivot.reindex(columns=compare_terms)
# Only subjects with ≥15 sections in Fall 2019
pivot = pivot[pivot['Fall 2019'] >= 15]
# Normalize to Fall 2019 = 100
baseline = pivot['Fall 2019']
pivot_norm = pivot.div(baseline, axis=0) * 100
pivot_norm = pivot_norm.round(1)
# Sort by Fall 2020 impact (biggest drop first)
pivot_norm = pivot_norm.sort_values('Fall 2020', ascending=True)

fig = go.Figure(go.Heatmap(
    z=pivot_norm.values,
    x=pivot_norm.columns.tolist(),
    y=pivot_norm.index.tolist(),
    colorscale=[[0, '#c0392b'], [0.5, '#fdfefe'], [1, '#27ae60']],
    zmid=100, zmin=50, zmax=150,
    text=pivot_norm.values.round(0).astype(int).astype(str),
    texttemplate='%{text}',
    textfont=dict(size=10),
    colorbar=dict(title='Index<br>(F19=100)', tickvals=[50, 75, 100, 125, 150]),
    hovertemplate='%{y} %{x}: %{z:.0f} (Fall 2019 = 100)<extra></extra>'))
fig.update_layout(template=TEMPLATE,
                  title="Subject Sections Indexed to Fall 2019 = 100",
                  height=max(450, len(pivot_norm) * 24),
                  xaxis=dict(side='top'), yaxis=dict(dtick=1))
charts['covid_subjects'] = chart_html(fig, 'covid_subjects')

# Count subjects that grew vs shrank
f20_vals = pivot_norm['Fall 2020']
subjects_shrank = int((f20_vals < 90).sum())
subjects_grew = int((f20_vals > 110).sum())
subjects_stable = int(len(f20_vals) - subjects_shrank - subjects_grew)

# --- Chart 4: Unassigned classrooms — permanent structural shift ---
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df_covid['term_name'], y=df_covid['pct_no_classroom'],
    marker_color=['#c0392b' if t in ('Spring 2020', 'Fall 2020', 'Spring 2021') else '#2471a3'
                  for t in df_covid['term_name']],
    hovertemplate='%{x}<br>%{y:.1f}% without classroom<extra></extra>'))
fig.add_vrect(x0='Spring 2020', x1='Spring 2021',
              fillcolor='rgba(192,57,43,0.12)', line_width=0)
fig.update_layout(template=TEMPLATE,
                  title="Sections Without Assigned Classroom (%)",
                  height=420, xaxis=dict(tickangle=-45, dtick=4),
                  yaxis_title="% of Sections",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts['covid_classrooms'] = chart_html(fig, 'covid_classrooms')

precovid_noroom = round(df_covid[df_covid['term_name'] == 'Fall 2019']['pct_no_classroom'].iloc[0], 1)
latest_noroom = round(df_covid['pct_no_classroom'].iloc[-1], 1)

# ===== NEW: Curriculum Evolution =====
df_first_offered = pd.read_sql_query("""
    SELECT c.subject, c.course_number, MIN(s.term_id) as first_term_id,
           MIN(s.term_name) as first_term_name,
           COUNT(DISTINCT c.term_id) as semesters_offered,
           MAX(s.term_id) as last_term_id,
           MAX(s.term_name) as last_term_name
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    GROUP BY c.subject, c.course_number
""", conn)

# New courses per year
df_first_offered['first_year'] = df_first_offered['first_term_id'].astype(str).str[:4].astype(int)
df_new_per_year = df_first_offered.groupby('first_year').size().reset_index(name='new_courses')

fig = px.bar(df_new_per_year, x='first_year', y='new_courses',
             title="New Courses Introduced Per Year",
             labels={'first_year': 'Year', 'new_courses': 'New Courses'},
             color='new_courses', color_continuous_scale='Teal')
fig.update_layout(template=TEMPLATE, height=400)
fig.update_coloraxes(showscale=False)
charts['new_courses'] = chart_html(fig, 'new_courses')

peak_new_year = int(df_new_per_year.loc[df_new_per_year['new_courses'].idxmax(), 'first_year'])
peak_new_count = int(df_new_per_year['new_courses'].max())

# Course longevity histogram
fig = px.histogram(df_first_offered, x='semesters_offered', nbins=40,
                   title="Course Longevity: How Many Semesters Is Each Course Offered?",
                   labels={'semesters_offered': 'Semesters Offered', 'count': 'Courses'},
                   color_discrete_sequence=[AUS_GOLD])
fig.update_layout(template=TEMPLATE, height=400)
charts['course_longevity'] = chart_html(fig, 'course_longevity')

one_sem_courses = int(len(df_first_offered[df_first_offered['semesters_offered'] == 1]))
total_unique_courses = int(len(df_first_offered))
one_sem_pct = round(one_sem_courses / total_unique_courses * 100, 1)
veteran_courses = int(len(df_first_offered[df_first_offered['semesters_offered'] >= 30]))

# Most consistently offered courses
df_consistent = df_first_offered.nlargest(20, 'semesters_offered').sort_values('semesters_offered', ascending=True)
df_consistent['label'] = df_consistent['subject'] + ' ' + df_consistent['course_number']

fig = px.bar(df_consistent, y='label', x='semesters_offered', orientation='h',
             title="Most Consistently Offered Courses (by Semesters Active)",
             labels={'label': '', 'semesters_offered': 'Semesters Offered'},
             color='semesters_offered', color_continuous_scale=['#d4e6f1', '#1a5276'])
fig.update_layout(template=TEMPLATE, height=550)
fig.update_coloraxes(showscale=False)
charts['most_consistent'] = chart_html(fig, 'most_consistent')

# Courses discontinued (appeared in 2005-2015 but not after 2020)
max_term_id = int(df_first_offered['last_term_id'].max())
df_discontinued = df_first_offered[
    (df_first_offered['first_term_id'].astype(int) < 201500) &
    (df_first_offered['last_term_id'].astype(int) < 202000) &
    (df_first_offered['semesters_offered'] >= 5)
].copy()
df_discontinued['last_year'] = df_discontinued['last_term_id'].astype(str).str[:4].astype(int)
df_disc_per_year = df_discontinued.groupby('last_year').size().reset_index(name='discontinued')

fig = px.bar(df_disc_per_year, x='last_year', y='discontinued',
             title="Courses Last Offered Per Year (Courses Not Seen After 2020)",
             labels={'last_year': 'Last Year Offered', 'discontinued': 'Courses'},
             color='discontinued', color_continuous_scale='OrRd')
fig.update_layout(template=TEMPLATE, height=400)
fig.update_coloraxes(showscale=False)
charts['courses_discontinued'] = chart_html(fig, 'courses_discontinued')

total_discontinued = int(len(df_discontinued))

# ===== NEW: Corequisites =====
df_coreqs = pd.read_sql_query("""
    SELECT DISTINCT c.subject as course_subject, c.course_number as course_num,
        d.subject as dep_subject, d.course_number as dep_num
    FROM course_dependencies d
    JOIN courses c ON c.crn = d.crn AND c.term_id = d.term_id
    WHERE d.dep_type = 'corequisite'
""", conn)

# Top corequisite connections
coreq_counter = Counter()
for _, row in df_coreqs.iterrows():
    pair = tuple(sorted([f"{row['course_subject']} {row['course_num']}",
                         f"{row['dep_subject']} {row['dep_num']}"]))
    coreq_counter[pair] += 1

top_coreqs = coreq_counter.most_common(20)
if top_coreqs:
    coreq_labels = [f"{p[0]}  \u2194  {p[1]}" for p, _ in top_coreqs]
    coreq_counts = [c for _, c in top_coreqs]
    fig = go.Figure(go.Bar(y=coreq_labels[::-1], x=coreq_counts[::-1], orientation='h',
        marker_color='#8e44ad',
        hovertemplate="%{y}: %{x} links<extra></extra>"))
    fig.update_layout(template=TEMPLATE, title="Most Common Corequisite Pairs",
                      height=550, xaxis_title="Occurrences", margin=dict(l=300))
    charts['coreq_top'] = chart_html(fig, 'coreq_top')

# Coreqs vs prereqs by department
df_dep_compare = pd.read_sql_query("""
    SELECT c.subject, d.dep_type, COUNT(DISTINCT d.subject || d.course_number) as dep_count
    FROM course_dependencies d
    JOIN courses c ON c.crn = d.crn AND c.term_id = d.term_id
    GROUP BY c.subject, d.dep_type
""", conn)
df_dep_pvt = df_dep_compare.pivot_table(index='subject', columns='dep_type', values='dep_count', fill_value=0).reset_index()
if 'prerequisite' in df_dep_pvt.columns and 'corequisite' in df_dep_pvt.columns:
    df_dep_pvt['total'] = df_dep_pvt['prerequisite'] + df_dep_pvt['corequisite']
    df_dep_pvt = df_dep_pvt.nlargest(20, 'total').sort_values('total', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_dep_pvt['subject'], x=df_dep_pvt['prerequisite'],
        name='Prerequisites', orientation='h', marker_color='#e74c3c'))
    fig.add_trace(go.Bar(y=df_dep_pvt['subject'], x=df_dep_pvt['corequisite'],
        name='Corequisites', orientation='h', marker_color='#8e44ad'))
    fig.update_layout(template=TEMPLATE, title="Prerequisites vs Corequisites by Department (Top 20)",
                      barmode='group', height=550, xaxis_title="Unique Required Courses",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    charts['coreq_vs_prereq'] = chart_html(fig, 'coreq_vs_prereq')

total_coreq_links = int(len(df_coreqs))

# ===== NEW: Course Attributes =====
df_attr_raw = pd.read_sql_query("""
    SELECT c.subject, c.course_number, c.attributes, s.term_name, s.term_id, COUNT(*) as sections
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE c.attributes != ''
    GROUP BY c.subject, c.course_number, c.attributes, s.term_id
""", conn)

attr_counts = Counter()
for _, row in df_attr_raw.iterrows():
    for attr in str(row['attributes']).split(', '):
        attr = attr.strip()
        if attr:
            attr_counts[attr] += row['sections']

df_attrs = pd.DataFrame(attr_counts.most_common(25), columns=['attribute', 'sections'])

fig = px.bar(df_attrs.sort_values('sections', ascending=True),
             y='attribute', x='sections', orientation='h',
             title="Top 25 Course Attributes / Gen-Ed Tags",
             labels={'sections': 'Total Section-Occurrences', 'attribute': ''},
             color='sections', color_continuous_scale='Viridis')
fig.update_layout(template=TEMPLATE, height=650, margin=dict(l=300))
fig.update_coloraxes(showscale=False)
charts['attributes_dist'] = chart_html(fig, 'attributes_dist')

# Attribute categories over time
def categorize_attr(attr):
    a = attr.lower()
    if 'science' in a and 'social' not in a: return 'Natural Sciences'
    if 'social science' in a: return 'Social Sciences'
    if 'math' in a or 'stat' in a: return 'Math/Statistics'
    if 'communication' in a or 'english' in a or 'language' in a: return 'Communication/English'
    if 'arabic' in a or 'heritage' in a or 'humanities' in a or 'history' in a: return 'Humanities/Arabic'
    if 'art' in a or 'literature' in a: return 'Arts & Literature'
    if 'preparatory' in a: return 'Preparatory'
    if 'internship' in a: return 'Internship'
    if 'elective' in a: return 'Elective'
    return 'Other'

attr_time_data = []
for _, row in df_attr_raw.iterrows():
    year = int(str(row['term_id'])[:4])
    for attr in str(row['attributes']).split(', '):
        attr = attr.strip()
        if attr:
            attr_time_data.append({'year': year, 'category': categorize_attr(attr), 'sections': row['sections']})

df_attr_time = pd.DataFrame(attr_time_data)
df_attr_time_agg = df_attr_time.groupby(['year', 'category'])['sections'].sum().reset_index()

cat_colors = {'Natural Sciences': '#27ae60', 'Social Sciences': '#e67e22', 'Math/Statistics': '#2471a3',
              'Communication/English': '#c0392b', 'Humanities/Arabic': '#8e44ad', 'Arts & Literature': '#1abc9c',
              'Preparatory': '#d68910', 'Internship': '#16a085', 'Elective': '#95a5a6', 'Other': '#bdc3c7'}

fig = go.Figure()
for cat in ['Natural Sciences', 'Social Sciences', 'Math/Statistics', 'Communication/English',
            'Humanities/Arabic', 'Arts & Literature', 'Preparatory', 'Internship', 'Elective']:
    df_c = df_attr_time_agg[df_attr_time_agg['category'] == cat].sort_values('year')
    if len(df_c) > 0:
        fig.add_trace(go.Scatter(x=df_c['year'], y=df_c['sections'],
            name=cat, mode='lines', stackgroup='one',
            line=dict(color=cat_colors.get(cat, '#95a5a6'), width=0.5)))
fig.update_layout(template=TEMPLATE, title="Gen-Ed / Course Attribute Categories Over Time",
                  height=500, xaxis_title="Year", yaxis_title="Tagged Sections",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)))
charts['attributes_over_time'] = chart_html(fig, 'attributes_over_time')

total_unique_attrs = len(attr_counts)

# ===== NEW: Typed enrollment restrictions =====
# AUSCrawl's typed `restrictions_json` is only populated for recently-fetched
# sections, but the raw `restrictions` text — present for ~90% of sections —
# is highly regular, so we parse it into typed include/exclude groups for full
# coverage (see analysis.restriction_text_groups).
df_restr = pd.read_sql_query("""
    SELECT sd.crn, sd.term_id, sd.restrictions
    FROM section_details sd
    WHERE sd.restrictions != '' AND sd.restrictions IS NOT NULL
""", conn)

restr_label_counts = Counter()
restr_section_courses = []   # (course, title) for sections gated to a specific major/college/program
sec_course_map = pd.read_sql_query(
    "SELECT DISTINCT crn, term_id, subject, course_number, title FROM courses", conn)
sec_lookup = {(r.crn, r.term_id): (f"{r.subject} {r.course_number}", r.title)
              for r in sec_course_map.itertuples()}

total_restricted = 0
selective_sections = 0   # sections gated by a specific college/major/program/field
for row in df_restr.itertuples():
    groups = analysis.restriction_text_groups(row.restrictions)
    if not groups:
        continue
    total_restricted += 1
    for g in groups:
        restr_label_counts[analysis.restriction_label(g)] += 1
    if analysis.restriction_text_is_selective(row.restrictions):
        selective_sections += 1
        course = sec_lookup.get((row.crn, row.term_id))
        if course:
            restr_section_courses.append(course)

# N9: typed include/exclude restriction breakdown.
# Level gates are near-universal (every undergrad course is "undergraduate-
# only") and would dwarf everything, so the chart focuses on the rules that
# narrow further; the selective share is reported in the narrative instead.
df_rlabels = pd.DataFrame(
    [(lbl, n) for lbl, n in restr_label_counts.most_common() if "Level" not in lbl],
    columns=["label", "count"])
df_rlabels["kind"] = df_rlabels["label"].apply(
    lambda l: "Must be (include)" if l.startswith("Must be") else "Must not be (exclude)")
fig = px.bar(df_rlabels.sort_values("count"), x="count", y="label", orientation="h",
             color="kind", color_discrete_map={"Must be (include)": "#2471a3",
                                                "Must not be (exclude)": "#c0392b"},
             title="Selective Enrollment Restrictions by Type",
             labels={"count": "Section-Term Occurrences", "label": "", "kind": ""})
fig.update_layout(template=TEMPLATE, height=460, margin=dict(l=180),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts["restrictions_typed"] = chart_html(fig, "restrictions_typed")
selective_section_pct = round(selective_sections / table_counts['section_details'] * 100, 1)

# N10: courses most often gated to a specific major / college / program
restr_course_counts = Counter(c for c, _ in restr_section_courses)
restr_titles = dict(restr_section_courses)
df_restricted_courses = pd.DataFrame(
    [(c, restr_titles.get(c, ""), n) for c, n in restr_course_counts.most_common(18)],
    columns=["course", "title", "sections"]).sort_values("sections")
fig = go.Figure(go.Bar(
    y=[f"{r.course} — {r.title[:34]}" for r in df_restricted_courses.itertuples()],
    x=df_restricted_courses["sections"], orientation="h", marker_color="#8e44ad",
    hovertemplate="%{y}<br>%{x} restricted section-terms<extra></extra>"))
fig.update_layout(template=TEMPLATE,
                  title="Courses Most Restricted by Major / College / Program",
                  height=max(450, len(df_restricted_courses) * 26),
                  xaxis_title="Restricted Section-Terms", margin=dict(l=320),
                  yaxis=dict(autorange="reversed"))
charts["restricted_courses"] = chart_html(fig, "restricted_courses")

restricted_pct = round(total_restricted / table_counts['section_details'] * 100, 1)
selective_restricted = int(len(set(c for c, _ in restr_section_courses)))

# ===========================================================================
# NEW COVERAGE CHARTS — section_instructors, prerequisite trees, catalog_detail
# ===========================================================================

# ----- Team Teaching (section_instructors) --------------------------------
# One row per (section, instructor); collapse to per-section instructor counts.
df_si_counts = pd.read_sql_query("""
    SELECT crn, term_id, COUNT(*) AS n FROM section_instructors GROUP BY crn, term_id
""", conn)

# N1: co-instruction rate over time
df_co_time = pd.read_sql_query("""
    SELECT s.term_name, s.term_id, COUNT(*) AS sections,
           SUM(CASE WHEN si.n > 1 THEN 1 ELSE 0 END) AS co_taught
    FROM (SELECT crn, term_id, COUNT(*) AS n FROM section_instructors GROUP BY crn, term_id) si
    JOIN semesters s ON s.term_id = si.term_id
    WHERE s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%'
    GROUP BY si.term_id ORDER BY si.term_id
""", conn)
df_co_time["co_pct"] = (df_co_time["co_taught"] / df_co_time["sections"] * 100).round(1)
fig = px.area(df_co_time, x="term_name", y="co_pct",
              title="Co-Taught Sections Over Time (% With 2+ Instructors)",
              labels={"co_pct": "% Co-Taught", "term_name": "Semester"},
              color_discrete_sequence=[AUS_GOLD])
fig.update_traces(line=dict(width=2))
fig.update_layout(template=TEMPLATE, height=420, xaxis=dict(tickangle=-45, dtick=4))
charts["co_time"] = chart_html(fig, "co_time")
co_overall_pct = round(df_si_counts["n"].gt(1).mean() * 100, 1)

# N2: co-teaching rate by department
df_co_dept = pd.read_sql_query("""
    SELECT subj.subject, COUNT(*) AS total,
           SUM(CASE WHEN si.n > 1 THEN 1 ELSE 0 END) AS co_taught
    FROM (SELECT crn, term_id, COUNT(*) AS n FROM section_instructors GROUP BY crn, term_id) si
    JOIN (SELECT DISTINCT crn, term_id, subject FROM courses) subj
      ON subj.crn = si.crn AND subj.term_id = si.term_id
    GROUP BY subj.subject HAVING total >= 150 ORDER BY 1.0 * co_taught / total DESC LIMIT 18
""", conn)
df_co_dept["co_pct"] = (df_co_dept["co_taught"] / df_co_dept["total"] * 100).round(1)
fig = px.bar(df_co_dept.sort_values("co_pct"), x="co_pct", y="subject", orientation="h",
             color="co_pct", color_continuous_scale="Purples",
             hover_data=["co_taught", "total"],
             title="Co-Teaching Rate by Department (Top 18)",
             labels={"co_pct": "% of Sections Co-Taught", "subject": ""})
fig.update_layout(template=TEMPLATE, height=520)
fig.update_coloraxes(showscale=False)
charts["co_dept"] = chart_html(fig, "co_dept")

# N3: most frequent co-teaching partnerships
df_si_all = pd.read_sql_query("""
    SELECT crn, term_id, name FROM section_instructors
    WHERE name != '' AND name != 'TBA'
""", conn)
pair_counter = Counter()
for (crn, term_id), grp in df_si_all.groupby(["crn", "term_id"]):
    names = sorted(set(grp["name"]))
    if len(names) < 2:
        continue
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair_counter[(names[i], names[j])] += 1
top_pairs = pair_counter.most_common(15)
if top_pairs:
    pair_labels = [f"{a}  +  {b}" for (a, b), _ in top_pairs]
    pair_counts = [n for _, n in top_pairs]
    fig = go.Figure(go.Bar(y=pair_labels[::-1], x=pair_counts[::-1], orientation="h",
        marker_color="#9b59b6",
        hovertemplate="%{y}<br>%{x} sections together<extra></extra>"))
    fig.update_layout(template=TEMPLATE, title="Most Frequent Co-Teaching Partnerships",
                      height=520, xaxis_title="Sections Taught Together", margin=dict(l=320))
    charts["co_pairs"] = chart_html(fig, "co_pairs")

# ----- Prerequisite logic & complexity (prerequisites_json) ----------------
df_sd_pre = pd.read_sql_query("""
    SELECT crn, term_id, prerequisites_json FROM section_details
    WHERE prerequisites_json NOT IN ('', '[]', 'null') AND prerequisites_json IS NOT NULL
""", conn)
pl_rows = []
for row in df_sd_pre.itertuples():
    tree = analysis.load_tree(row.prerequisites_json)
    if not tree:
        continue
    course = sec_lookup.get((row.crn, row.term_id))
    if not course:
        continue
    code, title = course
    pl_rows.append({
        "term_id": row.term_id, "year": int(str(row.term_id)[:4]),
        "subject": code.split(" ")[0], "course": code, "title": title,
        "shape": analysis.classify_prereq_shape(tree),
        "depth": analysis.tree_depth(tree),
        "mandatory": analysis.tree_mandatory_count(tree),
        "has_or": analysis.tree_has_or(tree),
        "concurrent": analysis.tree_has_concurrent(tree),
    })
df_pl = pd.DataFrame(pl_rows)
# Course-level view: latest tree per course (prereqs evolve over time).
df_pl_course = df_pl.sort_values("term_id").drop_duplicates("course", keep="last")

# N4: prerequisite shape distribution
shape_order = ["Single course", "All required (AND)", "Has alternatives (OR)", "Complex (nested AND/OR)"]
shape_colors = {"Single course": "#95a5a6", "All required (AND)": "#2471a3",
                "Has alternatives (OR)": "#27ae60", "Complex (nested AND/OR)": "#c0392b"}
df_shape = df_pl_course["shape"].value_counts().reindex(shape_order).fillna(0).reset_index()
df_shape.columns = ["shape", "courses"]
fig = px.bar(df_shape, x="shape", y="courses", color="shape",
             color_discrete_map=shape_colors,
             title="How Are Course Prerequisites Structured?",
             labels={"courses": "Unique Courses", "shape": ""})
fig.update_layout(template=TEMPLATE, height=430, showlegend=False)
charts["prereq_shape"] = chart_html(fig, "prereq_shape")
or_share = round((df_pl_course["has_or"]).mean() * 100, 1)

# N5: prerequisite nesting depth
df_depth = df_pl_course["depth"].value_counts().sort_index().reset_index()
df_depth.columns = ["depth", "courses"]
depth_names = {1: "1 — single course", 2: "2 — one AND/OR group",
               3: "3 — nested", 4: "4 — deeply nested", 5: "5 — deeply nested"}
df_depth["label"] = df_depth["depth"].map(lambda d: depth_names.get(d, str(d)))
fig = px.bar(df_depth, x="label", y="courses", color="depth",
             color_continuous_scale="Sunsetdark",
             title="Prerequisite Logic Nesting Depth",
             labels={"courses": "Unique Courses", "label": "Boolean Nesting Depth"})
fig.update_layout(template=TEMPLATE, height=430)
fig.update_coloraxes(showscale=False)
charts["prereq_depth"] = chart_html(fig, "prereq_depth")

# N6: toughest gateways — most mandatory (AND-ed) prerequisites
df_gate = df_pl_course.nlargest(18, "mandatory").sort_values("mandatory")
fig = go.Figure(go.Bar(
    y=[f"{r.course} — {str(r.title)[:30]}" for r in df_gate.itertuples()],
    x=df_gate["mandatory"], orientation="h", marker_color="#c0392b",
    hovertemplate="%{y}<br>%{x} mandatory prerequisites<extra></extra>"))
fig.update_layout(template=TEMPLATE,
                  title="Toughest Gateways: Most Mandatory Prerequisites",
                  height=560, xaxis_title="Courses That Must All Be Passed First",
                  margin=dict(l=320), yaxis=dict(autorange="reversed"))
charts["prereq_gateways"] = chart_html(fig, "prereq_gateways")
toughest = df_gate.iloc[-1]

# N7: curriculum flexibility — share of gated courses offering alternatives
df_flex = df_pl_course.groupby("subject").agg(
    courses=("course", "count"), with_or=("has_or", "sum")).reset_index()
df_flex = df_flex[df_flex["courses"] >= 8]
df_flex["flex_pct"] = (df_flex["with_or"] / df_flex["courses"] * 100).round(1)
df_flex = df_flex.sort_values("flex_pct", ascending=False).head(18).sort_values("flex_pct")
fig = px.bar(df_flex, x="flex_pct", y="subject", orientation="h",
             color="flex_pct", color_continuous_scale="Greens",
             hover_data=["with_or", "courses"],
             title="Share of Gated Courses Offering Alternative Paths (by Dept)",
             labels={"flex_pct": "% With OR-Alternatives", "subject": ""})
fig.update_layout(template=TEMPLATE, height=520)
fig.update_coloraxes(showscale=False)
charts["prereq_flex"] = chart_html(fig, "prereq_flex")

# N8: concurrent ("may be taken together") prerequisites over time
df_conc = df_pl.groupby("year").agg(
    trees=("course", "count"), conc=("concurrent", "sum")).reset_index()
df_conc["conc_pct"] = (df_conc["conc"] / df_conc["trees"] * 100).round(1)
fig = px.line(df_conc, x="year", y="conc_pct", markers=True,
              title="Prerequisites Allowing Concurrent Enrollment Over Time",
              labels={"conc_pct": "% of Prereq'd Sections", "year": "Year"},
              color_discrete_sequence=["#16a085"])
fig.update_traces(line=dict(width=2.5))
fig.update_layout(template=TEMPLATE, height=400)
charts["prereq_concurrent"] = chart_html(fig, "prereq_concurrent")
concurrent_overall = round(df_pl["concurrent"].mean() * 100, 1)

# ----- Degree-requirement mapping (catalog_detail.course_attributes) -------
df_cd = pd.read_sql_query("""
    SELECT subject, course_number, course_attributes
    FROM catalog_detail WHERE course_attributes != ''
""", conn)
program_courses = {}          # program -> set of courses (elective options)
course_programs = Counter()   # course -> number of distinct programs served
for row in df_cd.itertuples():
    code = f"{row.subject} {row.course_number}"
    progs = set()
    for tag in analysis.parse_attribute_tags(row.course_attributes):
        prog = analysis.attribute_program(tag)
        progs.add(prog)
        if analysis.attribute_role(tag) == "Elective":
            program_courses.setdefault(prog, set()).add(code)
    course_programs[code] = len(progs)

# N11: programs with the most elective course options
df_prog = pd.DataFrame(
    [(p, len(cs)) for p, cs in program_courses.items()],
    columns=["program", "options"]).nlargest(20, "options").sort_values("options")
fig = px.bar(df_prog, x="options", y="program", orientation="h",
             color="options", color_continuous_scale="Tealgrn",
             title="Programs With the Most Elective Course Options",
             labels={"options": "Distinct Elective Courses", "program": ""})
fig.update_layout(template=TEMPLATE, height=560, margin=dict(l=240))
fig.update_coloraxes(showscale=False)
charts["program_electives"] = chart_html(fig, "program_electives")

# N12: most reusable courses — serve the most degree programs
df_reuse = pd.DataFrame(course_programs.most_common(20), columns=["course", "programs"])
df_reuse_titles = pd.read_sql_query("""
    SELECT DISTINCT subject || ' ' || course_number AS course, title FROM courses
""", conn)
title_map = dict(zip(df_reuse_titles["course"], df_reuse_titles["title"]))
df_reuse["label"] = df_reuse["course"].apply(lambda c: f"{c} — {str(title_map.get(c, ''))[:30]}")
df_reuse = df_reuse.sort_values("programs")
fig = go.Figure(go.Bar(y=df_reuse["label"], x=df_reuse["programs"], orientation="h",
    marker_color="#16a085",
    hovertemplate="%{y}<br>serves %{x} programs<extra></extra>"))
fig.update_layout(template=TEMPLATE,
                  title="Most Reusable Courses Across Degree Programs",
                  height=560, xaxis_title="Distinct Degree Programs Served",
                  margin=dict(l=320), yaxis=dict(autorange="reversed"))
charts["reusable_courses"] = chart_html(fig, "reusable_courses")

conn.close()

# ---------------------------------------------------------------------------
# Build browsable course data JSON for the interactive table
# ---------------------------------------------------------------------------
conn2 = sqlite3.connect(DB_PATH)
df_browse = pd.read_sql_query("""
    SELECT c.subject, c.course_number, c.title, c.section, c.credits,
           c.instructor_name, c.days, c.start_time, c.end_time,
           c.classroom, c.seats_available, s.term_name
    FROM courses c
    JOIN semesters s ON c.term_id = s.term_id
    ORDER BY s.term_id DESC, c.subject, c.course_number
""", conn2)
# Take latest 5000 for the interactive table (keep it fast)
df_browse_recent = df_browse.head(5000)
browse_json = df_browse_recent.to_json(orient="records")

# Catalog browse
df_cat_browse = pd.read_sql_query("""
    SELECT c.subject, c.course_number,
           (SELECT x.title FROM courses x
            WHERE x.subject = c.subject AND x.course_number = c.course_number
              AND x.title != '' ORDER BY x.term_id DESC LIMIT 1) AS title,
           c.description, c.credit_hours, c.lecture_hours, c.lab_hours, c.department
    FROM catalog c ORDER BY c.subject, c.course_number
""", conn2)
catalog_json = df_cat_browse.to_json(orient="records")

# Latest title per course code — so the dependency explorer can show names.
df_titles = pd.read_sql_query("""
    SELECT subject || ' ' || course_number AS code, title, MAX(term_id)
    FROM courses WHERE title != ''
    GROUP BY subject, course_number
""", conn2)
course_titles = dict(zip(df_titles['code'], df_titles['title']))
course_titles_json = json.dumps(course_titles)

# ---- Course profiles (title, credits, offering history, instructors) -------
sem_name_map = {str(k): v for k, v in
                pd.read_sql_query("SELECT term_id, term_name FROM semesters", conn2).itertuples(index=False)}
prof = pd.read_sql_query(
    "SELECT subject||' '||course_number AS code, crn, term_id, credits, days, start_time FROM courses", conn2)
prof['term_id'] = prof['term_id'].astype(str)
prof['year'] = prof['term_id'].str[:4].astype(int)

def _top_per_course(df, key):
    """Most common non-empty value of `key` per course code."""
    s = df[df[key].notna() & (df[key].astype(str) != '')]
    if s.empty:
        return {}
    return (s.groupby(['code', key]).size().reset_index(name='n')
              .sort_values('n').drop_duplicates('code', keep='last')
              .set_index('code')[key].to_dict())

day_top = _top_per_course(prof, 'days')
time_top = _top_per_course(prof[prof['start_time'] != '12:00 am'], 'start_time')
cred_top = _top_per_course(prof, 'credits')
agg = prof.groupby('code').agg(nt=('term_id', 'nunique'),
                               first_tid=('term_id', 'min'), last_tid=('term_id', 'max'))
nsec = prof.drop_duplicates(['code', 'crn', 'term_id']).groupby('code').size()
yrs = prof.groupby('code')['year'].apply(lambda s: sorted(set(s.tolist())))

course_profiles = {}
for code, row in agg.iterrows():
    cr = cred_top.get(code)
    course_profiles[code] = {
        't': course_titles.get(code, ''),
        'cr': (float(cr) if cr is not None and not pd.isna(cr) else None),
        'ns': int(nsec.get(code, 0)), 'nt': int(row['nt']),
        'ft': sem_name_map.get(str(row['first_tid']), ''),
        'lt': sem_name_map.get(str(row['last_tid']), ''),
        'yrs': yrs.get(code, []), 'dy': day_top.get(code, ''), 'tm': time_top.get(code, ''),
    }

inst_pc = pd.read_sql_query("""
    SELECT subj.code AS code, si.name AS name, COUNT(*) AS c
    FROM section_instructors si
    JOIN (SELECT DISTINCT crn, term_id, subject||' '||course_number AS code FROM courses) subj
      ON subj.crn = si.crn AND subj.term_id = si.term_id
    WHERE si.name != '' AND si.name != 'TBA'
    GROUP BY subj.code, si.name
""", conn2)
ninst = inst_pc.groupby('code')['name'].nunique().to_dict()
for code, g in inst_pc.sort_values('c').groupby('code'):
    if code in course_profiles:
        top = g.tail(8).iloc[::-1]
        course_profiles[code]['ins'] = [[n, int(cc)] for n, cc in zip(top['name'], top['c'])]
        course_profiles[code]['ni'] = int(ninst.get(code, 0))

# Sections offered per academic year, per course (for the offering bar chart)
sec_year = (prof.drop_duplicates(['code', 'crn', 'term_id'])
            .groupby(['code', 'year']).size().reset_index(name='n'))
for code, g in sec_year.groupby('code'):
    if code in course_profiles:
        course_profiles[code]['yc'] = [[int(y), int(n)] for y, n in zip(g['year'], g['n'])]
course_profiles_json = json.dumps(course_profiles)

# ---- Latest prerequisite logic tree per course (Requirement roadmap) -------
pt = pd.read_sql_query("""
    SELECT c.code AS code, sd.prerequisites_json AS pj, sd.term_id AS term_id
    FROM section_details sd
    JOIN (SELECT DISTINCT crn, term_id, subject||' '||course_number AS code FROM courses) c
      ON c.crn = sd.crn AND c.term_id = sd.term_id
    WHERE sd.prerequisites_json NOT IN ('', '[]', 'null') AND sd.prerequisites_json IS NOT NULL
""", conn2)
pt['term_id'] = pt['term_id'].astype(str)
pt = pt.sort_values('term_id').drop_duplicates('code', keep='last')

def _compact_tree(n):
    """Slim a prereq tree for the client: short keys, drop the always-present
    'Undergraduate' level, omit default min_grade/concurrent."""
    if n.get("type") == "course":
        c = {"c": f"{n.get('subject', '')} {n.get('course_number', '')}".strip()}
        if n.get("min_grade"):
            c["g"] = n["min_grade"]
        if n.get("concurrent"):
            c["k"] = 1
        return c
    return {"o": n.get("type"), "x": [_compact_tree(x) for x in n.get("operands", [])]}

prereq_trees = {}
for r in pt.itertuples():
    tree = analysis.load_tree(r.pj)
    if tree:
        prereq_trees[r.code] = _compact_tree(tree)
prereq_trees_json = json.dumps(prereq_trees)

# ---- Degree-program -> courses map (Degree Explorer) -----------------------
prog_rows = pd.read_sql_query(
    "SELECT subject||' '||course_number AS code, course_attributes FROM catalog_detail WHERE course_attributes != ''", conn2)
program_courses = {}
for r in prog_rows.itertuples():
    for tag in analysis.parse_attribute_tags(r.course_attributes):
        program_courses.setdefault(analysis.attribute_program(tag), {}).setdefault(r.code, analysis.attribute_role(tag))
program_map = {p: sorted([c, role] for c, role in courses.items())
               for p, courses in program_courses.items() if len(courses) >= 3}
program_map_json = json.dumps(program_map)

# Reverse map: which programs each course counts toward (for the Course Explorer)
course_to_programs = {}
for prog, courses in program_courses.items():
    if len(courses) < 3:
        continue
    for code in courses:
        course_to_programs.setdefault(code, []).append(prog)
# Store [total_count, up-to-15 program names] so broad free-electives stay compact.
course_to_programs = {code: [len(progs), sorted(progs)[:15]] for code, progs in course_to_programs.items()}
course_to_programs_json = json.dumps(course_to_programs)

# Build compact dependency data for Course Dependency Explorer
dep_data = {}
for node in G.nodes():
    dep_data[node] = {
        'p': [{'c': pred, 'g': G.edges[pred, node].get('grade', '')}
              for pred in G.predecessors(node)],
        'n': list(G.successors(node))
    }
# Add corequisites from df_coreqs
for _, row in df_coreqs.iterrows():
    course = f"{row['course_subject']} {row['course_num']}"
    coreq = f"{row['dep_subject']} {row['dep_num']}"
    if course in dep_data:
        dep_data[course].setdefault('q', []).append(coreq)
    else:
        dep_data[course] = {'p': [], 'n': [], 'q': [coreq]}
dep_explorer_json = json.dumps(dep_data)

# Build instructor data for Instructor Career Explorer
df_inst_detail = pd.read_sql_query("""
    SELECT c.instructor_name, c.crn, c.subject, c.course_number, c.title, s.term_name, s.term_id
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE c.instructor_name != '' AND c.instructor_name != 'TBA'
    ORDER BY s.term_id
""", conn2)
df_inst_detail['term_id'] = df_inst_detail['term_id'].astype(str)
df_inst_detail['year'] = df_inst_detail['term_id'].str[:4].astype(int)

inst_data = {}
for name, group in df_inst_detail.groupby('instructor_name'):
    secs = group.drop_duplicates(['crn', 'term_id'])
    top_courses = (secs.groupby(['subject', 'course_number'])
                   .agg(title=('title', 'first'), times=('term_id', 'nunique'))
                   .reset_index().nlargest(10, 'times'))
    yr_counts = secs.groupby('year').size()
    subj_counts = secs.groupby('subject').size().sort_values(ascending=False)
    inst_data[name] = {
        't': int(len(secs)),
        's': sorted(group['subject'].unique().tolist()),
        'n': int(group['term_id'].nunique()),
        'f': group['term_name'].iloc[0],
        'l': group['term_name'].iloc[-1],
        'c': [[r['subject'], r['course_number'], r['title'], int(r['times'])] for _, r in top_courses.iterrows()],
        'a': [[int(y), int(n)] for y, n in yr_counts.items()],
        'sm': [[s, int(n)] for s, n in subj_counts.head(8).items()],
    }
inst_explorer_json = json.dumps(inst_data)

conn2.close()

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

# One JSON blob holding every chart spec, rendered lazily on the client.
# Escape "</" so the payload can't prematurely close its <script> tag.
chart_specs_json = json.dumps(chart_specs, separators=(",", ":")).replace("</", "<\\/")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VisualizeAUS — 20 Years of AUS Course Data</title>
<meta name="description" content="Interactive visualizations of 20 years of course data from the American University of Sharjah.">
<script>
// Apply the saved/system theme before first paint to avoid a flash.
(function () {{
  try {{
    var saved = localStorage.getItem('viz-theme');
    var dark = saved ? saved === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (dark) document.documentElement.setAttribute('data-theme', 'dark');
  }} catch (e) {{}}
}})();
</script>
<link rel="preconnect" href="https://cdn.plot.ly">
<script defer src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;500;600;700;800&family=Montserrat:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  color-scheme: light;
  --bg: #faf7f2;
  --bg-warm: #f3ede3;
  --bg-card: #ffffff;
  --bg-hero: #010508;
  --nav-bg: rgba(250, 247, 242, 0.9);
  --border: #e2ddd4;
  --border-light: #ece8e0;
  --accent: #9e3223;
  --accent-light: #b84a3a;
  --accent-bg: #fdf5f4;
  --text: #1a1a1a;
  --text-secondary: #4a4a4a;
  --text-muted: #7a7a7a;
  --text-light: #999;
  --text-on-dark: #f0ece4;
  --cream: #e0d0af;
  --cream-light: #f0e8d4;
  /* semantic pill tints (prereq / coreq / unlocks / degree program) */
  --pre-bg: #fdf5f4;  --pre-border: #f0dbd8;
  --co-bg: #f5ecfb;   --co-border: #ddc6ec;
  --next-bg: #e9f6ea; --next-border: #aad8ac;
  --prog-bg: #fbf3e1; --prog-border: #e7d6ab;
  /* chart surface — charts re-theme to match (transparent bg inherits this) */
  --chart-surface: #ffffff;
  --shadow-sm: 0 1px 2px rgba(40, 30, 20, 0.04), 0 2px 6px rgba(40, 30, 20, 0.05);
  --shadow-md: 0 1px 2px rgba(40, 30, 20, 0.05), 0 6px 18px rgba(40, 30, 20, 0.08);
  --radius: 12px;
  --radius-sm: 8px;
  --font-display: 'Raleway', sans-serif;
  --font-body: 'Montserrat', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --max-width: 1100px;
}}

:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #15110d;
  --bg-warm: #1e1a15;
  --bg-card: #211c17;
  --bg-hero: #0a0806;
  --nav-bg: rgba(21, 17, 13, 0.88);
  --border: #38322b;
  --border-light: #2b2620;
  --accent: #e3765f;
  --accent-light: #ee8b74;
  --accent-bg: #2e1e1a;
  --text: #efe9df;
  --text-secondary: #c8c0b2;
  --text-muted: #918879;
  --text-light: #6e675c;
  --text-on-dark: #f0ece4;
  --cream: #e0d0af;
  --cream-light: #3b3322;
  --pre-bg: #2e1e1a;  --pre-border: #4d3029;
  --co-bg: #251d2d;   --co-border: #443454;
  --next-bg: #16261b; --next-border: #2e4b34;
  --prog-bg: #2b2415; --prog-border: #4c4027;
  --chart-surface: #1d1812;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3), 0 2px 6px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 2px 6px rgba(0, 0, 0, 0.4), 0 8px 22px rgba(0, 0, 0, 0.5);
}}

html {{ scroll-behavior: smooth; }}

body {{
  font-family: var(--font-body);
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  font-size: 16px;
  font-weight: 400;
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  overflow-x: hidden;
  transition: background-color 0.3s ease, color 0.3s ease;
}}

/* Respect reduced-motion: drop transitions/animations for users who ask. */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }}
}}

/* Keyboard focus visibility (never remove focus rings). */
a:focus-visible, button:focus-visible, input:focus-visible,
select:focus-visible, .tab:focus-visible, .cx-chip:focus-visible,
.example-chip:focus-visible, [tabindex]:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: var(--radius-sm);
}}
/* The main region is only a skip-link target — don't ring the whole page. */
#main-content:focus, #main-content:focus-visible {{ outline: none; }}

/* Skip-to-content link: hidden until focused via keyboard. */
.skip-link {{
  position: absolute;
  left: 1rem;
  top: -4rem;
  z-index: 1000;
  background: var(--accent);
  color: #fff;
  padding: 0.6rem 1.1rem;
  border-radius: var(--radius-sm);
  font-weight: 600;
  text-decoration: none;
  box-shadow: var(--shadow-md);
  transition: top 0.2s ease;
}}
.skip-link:focus {{ top: 1rem; }}

/* Autocomplete suggestion options. */
.sugg-opt:hover {{ background: var(--bg-warm); }}
/* Keyboard-active option needs a clear, distinct indicator (sole cue). */
.sugg-opt[aria-selected="true"] {{ background: var(--accent-bg); box-shadow: inset 3px 0 0 var(--accent); }}

/* Cleaner line breaks: balance short headings, avoid orphans in body copy. */
h1, h2, h3, .section-header h2 {{ text-wrap: balance; }}
.section-header p, .explanation, .insight, .hero .subtitle {{ text-wrap: pretty; }}

/* ===================== HERO ===================== */
.hero {{
  background: var(--bg-hero);
  color: var(--text-on-dark);
  padding: 6rem 2rem 5rem;
  text-align: center;
}}

.hero-inner {{
  max-width: var(--max-width);
  margin: 0 auto;
}}

.hero-eyebrow {{
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--cream);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 1.25rem;
}}

.hero h1 {{
  font-family: var(--font-display);
  font-size: clamp(2.5rem, 6vw, 4rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.15;
  color: #fff;
  margin-bottom: 1.25rem;
}}

.hero .subtitle {{
  font-size: 1.1rem;
  color: rgba(240, 236, 228, 0.7);
  font-weight: 300;
  max-width: 600px;
  margin: 0 auto 3rem;
  line-height: 1.75;
}}

.stats-row {{
  display: flex;
  justify-content: center;
  gap: 0;
  background: rgba(255,255,255,0.05);
  border-radius: var(--radius);
  max-width: 680px;
  margin: 0 auto;
  border: 1px solid rgba(255,255,255,0.08);
}}

.stat {{
  flex: 1;
  padding: 1.5rem 1rem;
  text-align: center;
}}

.stat + .stat {{
  border-left: 1px solid rgba(255,255,255,0.08);
}}

.stat-value {{
  font-family: var(--font-mono);
  font-size: 1.6rem;
  font-weight: 500;
  color: var(--cream);
  line-height: 1;
}}

.stat-label {{
  font-size: 0.7rem;
  color: rgba(240,236,228,0.5);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 0.4rem;
  font-weight: 500;
}}

/* ===================== NAV ===================== */
nav {{
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--nav-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  transition: background-color 0.3s ease, border-color 0.3s ease;
}}

nav .nav-bar {{
  max-width: var(--max-width);
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0 2rem;
}}

nav .nav-scroll {{
  position: relative;
  flex: 1;
  min-width: 0;
  display: flex;
}}

nav .nav-inner {{
  display: flex;
  align-items: center;
  gap: 0;
  flex: 1;
  min-width: 0;
  overflow-x: auto;
  scrollbar-width: none;
  scroll-behavior: smooth;
}}

nav .nav-inner::-webkit-scrollbar {{ display: none; }}

/* edge fades + chevrons signalling the nav can scroll horizontally */
nav .nav-edge {{
  position: absolute;
  top: 0;
  bottom: 0;
  width: 36px;
  display: flex;
  align-items: center;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s ease;
  z-index: 2;
  color: var(--accent);
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 1.1rem;
}}
nav .nav-edge.left {{ left: 0; justify-content: flex-start; padding-left: 3px; background: linear-gradient(to right, var(--nav-bg) 45%, rgba(0, 0, 0, 0)); }}
nav .nav-edge.right {{ right: 0; justify-content: flex-end; padding-right: 3px; background: linear-gradient(to left, var(--nav-bg) 45%, rgba(0, 0, 0, 0)); }}
nav .nav-edge.left::after {{ content: '\\2039'; }}
nav .nav-edge.right::after {{ content: '\\203A'; }}
nav .nav-scroll.more-left .nav-edge.left {{ opacity: 1; }}
nav .nav-scroll.more-right .nav-edge.right {{ opacity: 1; }}

nav a {{
  color: var(--text-muted);
  text-decoration: none;
  font-size: 0.82rem;
  font-weight: 500;
  white-space: nowrap;
  transition: color 0.2s ease;
  padding: 0.9rem 1rem;
  border-bottom: 2px solid transparent;
}}

nav a:hover {{ color: var(--text); }}
nav a.active {{ color: var(--accent); border-bottom-color: var(--accent); }}

nav .nav-brand {{
  font-family: var(--font-display);
  font-weight: 700;
  color: var(--text);
  font-size: 0.95rem;
  padding-right: 1.25rem;
  margin-right: 0.5rem;
  border-right: 1px solid var(--border);
  flex-shrink: 0;          /* pinned outside the scroller — always visible */
  white-space: nowrap;
  text-decoration: none;
}}

.theme-toggle {{
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text-secondary);
  cursor: pointer;
  transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease, transform 0.1s ease;
}}
.theme-toggle:hover {{ color: var(--accent); border-color: var(--accent); }}
.theme-toggle:active {{ transform: scale(0.92); }}
.theme-toggle svg {{ width: 18px; height: 18px; display: block; }}
.theme-toggle .moon {{ display: none; }}
:root[data-theme="dark"] .theme-toggle .sun {{ display: none; }}
:root[data-theme="dark"] .theme-toggle .moon {{ display: block; }}

/* ===================== SECTIONS ===================== */
.container {{
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 0 2rem;
}}

section {{
  padding: 4.5rem 0;
}}

section + section {{
  border-top: 1px solid var(--border);
}}

.section-header {{
  margin-bottom: 2.5rem;
  max-width: 680px;
}}

.section-header .section-num {{
  font-family: var(--font-mono);
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--accent);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
}}

.section-header h2 {{
  font-family: var(--font-display);
  font-size: clamp(2rem, 4vw, 2.75rem);
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--text);
  margin-bottom: 0.75rem;
  line-height: 1.2;
}}

.section-header p {{
  color: var(--text-secondary);
  font-size: 1rem;
  line-height: 1.75;
}}

/* ===================== CHARTS ===================== */
.chart-container {{
  background: var(--chart-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
  transition: box-shadow 0.3s ease, border-color 0.3s ease;
}}

.chart-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}}

/* ===================== EXPLANATIONS ===================== */
.explanation {{
  background: var(--accent-bg);
  border: 1px solid var(--pre-border);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 1rem 1.25rem;
  margin: -0.5rem 0 2rem;
  font-size: 0.92rem;
  line-height: 1.7;
  color: var(--text-secondary);
}}

.explanation strong {{
  color: var(--accent);
  font-weight: 600;
}}

/* ===================== INSIGHT (key stats) ===================== */
.insight {{
  background: var(--bg-warm);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1.25rem 1.5rem;
  margin: 1.5rem 0;
  font-size: 0.92rem;
  line-height: 1.7;
  color: var(--text-secondary);
}}

.insight strong {{
  color: var(--text);
  font-weight: 600;
}}

/* ===================== TABLES ===================== */
.table-wrapper {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin: 1.5rem 0;
}}

.table-controls {{
  display: flex;
  gap: 0.75rem;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border);
  background: var(--bg-warm);
  flex-wrap: wrap;
}}

.table-controls input, .table-controls select {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 0.55rem 0.9rem;
  border-radius: var(--radius-sm);
  font-family: var(--font-body);
  font-size: 0.85rem;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}}

.table-controls input:focus, .table-controls select:focus {{
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(158,50,35,0.08);
}}

.table-controls input::placeholder {{ color: var(--text-light); }}
.table-controls input {{ flex: 1; min-width: 200px; }}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
}}

thead th {{
  background: var(--bg-warm);
  padding: 0.75rem 1rem;
  text-align: left;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  position: sticky;
  top: 0;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  transition: color 0.2s;
  font-family: var(--font-mono);
}}

thead th:hover {{ color: var(--accent); }}

tbody tr {{
  border-bottom: 1px solid var(--border-light);
  transition: background 0.15s;
}}

tbody tr:hover {{ background: var(--bg-warm); }}

tbody td {{
  padding: 0.6rem 1rem;
  color: var(--text-secondary);
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.table-scroll {{
  max-height: 500px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}}

.table-info {{
  padding: 0.75rem 1.25rem;
  border-top: 1px solid var(--border-light);
  font-size: 0.78rem;
  color: var(--text-light);
  font-family: var(--font-mono);
}}

/* ===================== PREREQ CHAINS ===================== */
.chains-title {{
  font-family: var(--font-display);
  color: var(--text);
  margin: 2.5rem 0 1rem;
  font-size: 1.25rem;
  font-weight: 700;
}}

.chains-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}}

.chain {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1rem 1.25rem;
}}

.chain-header {{
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--accent);
  margin-bottom: 0.75rem;
}}

.chain-step {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.3rem 0;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  color: var(--text);
}}

.chain-step-num {{
  min-width: 1.4rem;
  height: 1.4rem;
  border-radius: 50%;
  background: var(--accent-bg);
  color: var(--accent);
  font-size: 0.65rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}}

.chain-connector {{
  width: 1px;
  height: 0.4rem;
  background: var(--border);
  margin-left: 0.65rem;
}}

/* ===================== TABS ===================== */
.tabs {{
  display: flex;
  gap: 0;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
}}

.tab {{
  padding: 0.75rem 1.5rem;
  cursor: pointer;
  color: var(--text-muted);
  font-size: 0.85rem;
  font-weight: 500;
  border-bottom: 2px solid transparent;
  transition: color 0.2s, border-color 0.2s;
}}

.tab:hover {{ color: var(--text); }}
.tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}

.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* ===================== EXPLORER ===================== */
.example-chip {{
  background: var(--bg-warm);
  border: 1px solid var(--border);
  padding: 0.3rem 0.8rem;
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 0.82rem;
  color: var(--accent);
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}}
.example-chip:hover {{ background: var(--accent-bg); border-color: var(--accent); }}

.cx-section {{ margin-bottom: 1.5rem; }}
.cx-label {{
  font-family: var(--font-mono); font-size: 0.74rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.6rem;
}}
.cx-chip {{
  display: inline-block; padding: 0.3rem 0.7rem; border-radius: var(--radius-sm);
  font-size: 0.82rem; cursor: pointer; border: 1px solid var(--border); background: var(--bg-warm);
  transition: border-color 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease;
}}
.cx-chip .mono {{ font-family: var(--font-mono); font-weight: 500; }}
.cx-chip .nm {{ color: var(--text-muted); }}
/* semantic pill tints — prereq / coreq / unlocks / degree program */
.cx-chip.pre, .cx-leaf.pre {{ background: var(--pre-bg); border-color: var(--pre-border); }}
.cx-chip.co {{ background: var(--co-bg); border-color: var(--co-border); }}
.cx-chip.next {{ background: var(--next-bg); border-color: var(--next-border); }}
.cx-chip.prog {{ background: var(--prog-bg); border-color: var(--prog-border); }}
.cx-chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
/* honest framing caption for the degree explorer */
.cx-note {{
  font-size: 0.82rem; color: var(--text-secondary); background: var(--bg-warm);
  border-left: 3px solid var(--accent); padding: 0.65rem 0.9rem;
  border-radius: var(--radius-sm); margin-bottom: 1.4rem; line-height: 1.55;
  text-wrap: pretty;
}}
/* scrollable chip pool with a bottom fade + hint that signals "more below" */
.cx-scrollbox {{ position: relative; }}
.cx-scrollbox .cx-chips.scrolly {{
  max-height: 320px; overflow-y: auto; padding: 0.1rem 0.1rem 0.5rem;
  scrollbar-width: thin; scrollbar-color: var(--border) transparent;
}}
.cx-scrollbox .cx-fade {{
  position: absolute; left: 0; right: 0; bottom: 0; height: 42px;
  background: linear-gradient(to top, var(--bg-card), rgba(0, 0, 0, 0));
  pointer-events: none; opacity: 0; transition: opacity 0.2s ease;
  border-radius: 0 0 var(--radius-sm) var(--radius-sm);
}}
.cx-scrollbox .cx-more-hint {{
  position: absolute; right: 10px; bottom: 7px; font-family: var(--font-mono);
  font-size: 0.66rem; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--text-muted); pointer-events: none; opacity: 0;
  transition: opacity 0.2s ease;
}}
.cx-scrollbox.more .cx-fade, .cx-scrollbox.more .cx-more-hint {{ opacity: 1; }}
/* offering timeline strip */
.cx-strip {{ display: flex; gap: 2px; margin: 0.4rem 0 0.2rem; }}
.cx-strip .cell {{ width: 9px; height: 22px; border-radius: 2px; background: var(--border-light); }}
.cx-strip .cell.on {{ background: var(--accent); }}
/* AND/OR requirement tree */
.cx-tree {{ font-size: 0.85rem; }}
.cx-node {{ margin: 0.2rem 0; }}
.cx-op {{
  display: inline-block; font-family: var(--font-mono); font-size: 0.66rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em; padding: 0.1rem 0.45rem; border-radius: 4px;
  color: #fff; vertical-align: middle;
}}
.cx-op.and {{ background: #2471a3; }}
.cx-op.or {{ background: #1f8a4c; }}
.cx-kids {{ margin-left: 0.9rem; padding-left: 0.9rem; border-left: 2px solid var(--border); }}
.cx-leaf {{
  display: inline-block; margin: 0.15rem 0; padding: 0.2rem 0.6rem; border-radius: var(--radius-sm);
  font-size: 0.82rem; cursor: pointer; background: var(--pre-bg); border: 1px solid var(--pre-border);
  transition: border-color 0.15s ease, transform 0.1s ease;
}}
.cx-leaf .mono {{ font-family: var(--font-mono); font-weight: 500; }}
.cx-leaf .nm {{ color: var(--text-muted); }}
.cx-leaf:hover, .cx-chip:hover {{ border-color: var(--accent); }}
.cx-chip:active, .cx-leaf:active, .example-chip:active {{ transform: scale(0.96); }}
.cx-leaf .exp {{ color: var(--accent); font-weight: 700; margin-left: 0.3rem; }}

/* live table-summary mini bars (cheap, redraw on every keystroke) */
.mini-summary {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 0.9rem 1.1rem; margin-bottom: 1rem;
}}
.mini-bars {{ font-size: 0.8rem; }}
.mb-row {{ display: flex; align-items: center; gap: 0.6rem; margin: 0.2rem 0; }}
.mb-label {{
  width: 80px; flex-shrink: 0; text-align: right; font-family: var(--font-mono);
  color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.mb-track {{ flex: 1; background: var(--border-light); border-radius: 3px; height: 14px; overflow: hidden; }}
.mb-fill {{ display: block; height: 100%; border-radius: 3px; transition: width 0.35s ease; }}
.mb-val {{ width: 46px; flex-shrink: 0; font-family: var(--font-mono); color: var(--text-muted); }}

/* ===================== FOOTER ===================== */
footer {{
  text-align: center;
  padding: 3.5rem 2rem;
  color: var(--text-muted);
  font-size: 0.85rem;
  border-top: 1px solid var(--border);
  background: var(--bg-warm);
}}

footer a {{ color: var(--accent); text-decoration: none; }}
footer a:hover {{ text-decoration: underline; }}
footer p {{ margin-top: 0.3rem; }}

/* ===================== RESPONSIVE ===================== */

/* Plotly charts must not blow out their container */
.chart-container .plotly-graph-div,
.chart-container .js-plotly-plot {{
  max-width: 100%;
  overflow-x: auto;
}}

/* Tables always scrollable horizontally */
.table-scroll {{
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}}

@media (max-width: 768px) {{
  /* Layout */
  .container {{ padding: 0 0.75rem; }}
  section {{ padding: 2.5rem 0; }}

  /* Hero */
  .hero {{ padding: 3.5rem 1rem 3rem; }}
  .hero .subtitle {{ font-size: 0.95rem; margin-bottom: 2rem; }}
  .stats-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    max-width: 100%;
    border-radius: var(--radius-sm);
  }}
  .stat {{ padding: 0.75rem 0.5rem; }}
  .stat + .stat {{ border-left: 1px solid rgba(255,255,255,0.08); }}
  /* Third row item wraps to second row */
  .stat:nth-child(4) {{ border-top: 1px solid rgba(255,255,255,0.08); }}
  .stat:nth-child(5) {{ border-top: 1px solid rgba(255,255,255,0.08); }}
  .stat-value {{ font-size: 1.1rem; }}
  .stat-label {{ font-size: 0.6rem; letter-spacing: 0.08em; }}

  /* Nav */
  nav .nav-inner {{ padding: 0 0.5rem; }}
  nav .nav-brand {{ font-size: 0.82rem; padding-right: 0.75rem; margin-right: 0.25rem; }}
  nav a {{ padding: 0.7rem 0.5rem; font-size: 0.72rem; }}

  /* Section headers */
  .section-header {{ margin-bottom: 1.5rem; }}
  .section-header .section-num {{ font-size: 0.72rem; }}

  /* Charts */
  .chart-row {{ grid-template-columns: 1fr; gap: 1rem; }}
  .chart-container {{ padding: 0.5rem; margin-bottom: 1rem; }}
  .chart-container .plotly-graph-div {{ min-width: 0 !important; }}

  /* Explanations */
  .explanation {{ padding: 0.75rem 1rem; font-size: 0.85rem; margin: -0.25rem 0 1.5rem; }}

  /* Tables */
  .table-controls {{ padding: 0.75rem; gap: 0.5rem; }}
  .table-controls input {{ min-width: 0; flex: 1; font-size: 0.8rem; padding: 0.5rem 0.7rem; }}
  .table-controls select {{ font-size: 0.8rem; padding: 0.5rem 0.7rem; }}
  thead th {{ padding: 0.5rem 0.6rem; font-size: 0.6rem; }}
  tbody td {{ padding: 0.45rem 0.6rem; font-size: 0.78rem; max-width: 180px; }}
  .table-info {{ padding: 0.5rem 0.75rem; font-size: 0.7rem; }}

  /* Tabs */
  .tabs {{ overflow-x: auto; scrollbar-width: none; -webkit-overflow-scrolling: touch; }}
  .tabs::-webkit-scrollbar {{ display: none; }}
  .tab {{ padding: 0.6rem 1rem; font-size: 0.8rem; white-space: nowrap; flex-shrink: 0; }}

  /* Prerequisite chains */
  .chains-grid {{ grid-template-columns: 1fr; }}
  .chains-title {{ font-size: 1.05rem; margin-top: 1.5rem; }}

  /* Footer */
  footer {{ padding: 2rem 1rem; font-size: 0.8rem; }}

  /* Insight boxes */
  .insight {{ padding: 1rem; font-size: 0.85rem; }}
}}

</style>
</head>
<body>

<a class="skip-link" href="#main-content">Skip to content</a>

<!-- Hero -->
<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">American University of Sharjah</div>
    <h1>VisualizeAUS</h1>
    <p class="subtitle">
      Twenty years of AUS course data, open to explore. Every section,
      instructor, prerequisite, and schedule from 2005 to 2026.
    </p>
    <div class="stats-row">
      <div class="stat">
        <div class="stat-value">{table_counts['courses']:,}</div>
        <div class="stat-label">Sections</div>
      </div>
      <div class="stat">
        <div class="stat-value">{table_counts['semesters']}</div>
        <div class="stat-label">Semesters</div>
      </div>
      <div class="stat">
        <div class="stat-value">{table_counts['instructors']:,}</div>
        <div class="stat-label">Instructors</div>
      </div>
      <div class="stat">
        <div class="stat-value">{table_counts['course_dependencies']:,}</div>
        <div class="stat-label">Dependencies</div>
      </div>
      <div class="stat">
        <div class="stat-value">{len(charts)}</div>
        <div class="stat-label">Charts</div>
      </div>
    </div>
  </div>
</div>

<!-- Navigation -->
<nav>
  <div class="nav-bar">
    <a href="#" class="nav-brand">VisualizeAUS</a>
    <div class="nav-scroll" id="nav-scroll">
      <div class="nav-inner" id="nav-inner">
      <a href="#growth">Growth</a>
      <a href="#subjects">Subjects</a>
      <a href="#levels">Levels</a>
      <a href="#instructors">Instructors</a>
      <a href="#team-teaching">Team Teaching</a>
      <a href="#modality">Modality</a>
      <a href="#covid">COVID</a>
      <a href="#schedule">Schedule</a>
      <a href="#curriculum">Curriculum</a>
      <a href="#prerequisites">Prerequisites</a>
      <a href="#grades">Grades</a>
      <a href="#attributes">Attributes</a>
      <a href="#catalog">Catalog</a>
      <a href="#enrollment">Enrollment</a>
      <a href="#browse">Browse</a>
      </div>
      <span class="nav-edge left" aria-hidden="true"></span>
      <span class="nav-edge right" aria-hidden="true"></span>
    </div>
    <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle dark mode" title="Toggle dark mode">
      <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"></path></svg>
      <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
    </button>
  </div>
</nav>

<main class="container" id="main-content" tabindex="-1">

<!-- 1. Growth -->
<section id="growth">
  <div class="section-header">
    <div class="section-num">01 &mdash; University Growth</div>
    <h2>Two Decades of Expansion</h2>
    <p>How has AUS grown its course offerings from 2005 to 2026?</p>
  </div>
  <div class="chart-container">{charts['growth']}</div>
  <div class="explanation">
    Each dot is one semester. <strong>Red dots are Fall</strong>, blue are Spring, green are Summer. The dashed line is the overall trend. A regular semester went from about 1,100 sections in 2005 to nearly 2,000 in 2025, a <strong>{growth_pct:.0f}% increase</strong>. The busiest was <strong>{peak_sem}</strong> at {peak_val:,} sections. Summer terms are much smaller (200-400 sections), which is why they sit in the lower cluster.
  </div>
  <div class="chart-container">{charts['courses_vs_sections']}</div>
  <div class="explanation">
    The blue line is total sections offered; the red line is unique courses. Both climb, but sections climb faster. So AUS isn't just adding new courses, it's running <strong>more sections of the courses it already has</strong> to fit more students.
  </div>
</section>

<!-- 2. Subjects -->
<section id="subjects">
  <div class="section-header">
    <div class="section-num">02 &mdash; Subject Analysis</div>
    <h2>What Does AUS Teach?</h2>
    <p>{table_counts['subjects']} subject areas spanning engineering, sciences, arts, and humanities.</p>
  </div>
  <div class="chart-container">{charts['subjects_bar']}</div>
  <div class="explanation">
    Mathematics (MTH) runs the most sections of any subject, over 5,500 across 20 years, because almost every student takes several math courses. Civil (CVE), Mechanical (MCE), and Electrical Engineering (ELE) come next. AUS is an engineering school at heart, and the subject list shows it.
  </div>
  <div class="chart-container">{charts['subject_lines']}</div>
  <div class="explanation">
    The <strong>top 10 subjects</strong>, semester by semester. Most hold steady or grow. The engineering subjects tend to rise together, which usually points to programs expanding in step.
  </div>
  <div class="chart-container">{charts['subject_heatmap']}</div>
  <div class="explanation">
    Each cell is the number of sections a subject offered in a given year. <strong>Darker red means more sections.</strong> The growth is easy to spot: most rows get darker from left to right.
  </div>
</section>

<!-- 3. Academic Levels -->
<section id="levels">
  <div class="section-header">
    <div class="section-num">03 &mdash; Academic Levels</div>
    <h2>Undergraduate, Graduate, and Beyond</h2>
    <p>AUS offers courses across {len(level_order)} academic levels, from undergraduate to doctorate.</p>
  </div>
  <div class="chart-container">{charts['levels_over_time']}</div>
  <div class="explanation">
    Academic levels stacked over 20 years. <strong>Undergraduate sections dominate at {undergrad_pct:.0f}%</strong> of everything offered. Graduate courses have been here since 2005; the <strong>Doctorate program launched in 2019</strong> and the Achievement Academy in 2011. Every level has grown, but it's the undergraduate base that keeps expanding.
  </div>
  <div class="chart-container">{charts['levels_dist']}</div>
  <div class="explanation">
    Across all semesters, undergraduate education dominates by a wide margin. Graduate is a distant second at {grad_total:,} sections. The Doctorate and Achievement Academy are smaller, but both are growing.
  </div>
  <div class="chart-container">{charts['levels_by_subject']}</div>
  <div class="explanation">
    Which subjects teach across more than one level? The ones at the top have the largest share of graduate sections. MBA and several engineering programs carry a real graduate load, while MTH and PHY stay almost entirely undergraduate.
  </div>
</section>

<!-- 4. Instructors -->
<section id="instructors">
  <div class="section-header">
    <div class="section-num">04 &mdash; Instructor Analysis</div>
    <h2>The Teaching Workforce</h2>
    <p>{table_counts['instructors']:,} unique instructors have taught at AUS since 2005.</p>
  </div>
  <div class="chart-container">{charts['instructors']}</div>
  <div class="explanation">
    The most prolific instructor has taught <strong>nearly 500 sections</strong> over their career here. Color shows how many semesters someone has been active: darker blue is a longer tenure.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['tenure']}</div>
    <div class="chart-container">{charts['active_instructors']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> Most instructors don't stay long; the histogram piles up at 1-5 semesters. Even so, a sizeable group has taught for 20+ semesters (10+ years). <strong>Right:</strong> Active instructors per semester have grown from about 300 in 2005 to over 500 lately.
  </div>
  <div class="chart-container">{charts['instructor_diversity']}</div>
  <div class="explanation">
    Each bubble is an instructor. X is total sections taught, Y is how many distinct subjects they teach, and bubble size is tenure. <strong>The upper-right is prolific and versatile:</strong> lots of sections across several subjects. Most people sit in the lower-left (a few sections, 1-2 subjects), and a handful of outliers stand well apart.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['tba_rate']}</div>
    <div class="chart-container">{charts['instructor_workload']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> The TBA rate is the share of sections each semester with no instructor assigned at scrape time. A high rate in recent semesters usually just means staffing wasn't final yet. <strong>Right:</strong> Sections per instructor per semester barely moves, sitting around 3-4. Workloads stay fairly even.
  </div>
  <div class="chart-container">{charts['instructor_turnover']}</div>
  <div class="explanation">
    Green bars are new hires (first semester teaching); red bars are departures (last semester before they vanish from the data). Of {total_ever:,} instructors who have ever taught here, {still_active:,} are still active. Fall always brings in more new faculty than Spring, which tracks the annual hiring cycle.
  </div>
  <div class="chart-container">{charts['instructor_retention']}</div>
  <div class="explanation">
    Survival curves by hiring cohort: of the instructors hired in a given period, what share are still teaching N years later? The sharpest drop comes in the first year or two. Those who make it past about five years tend to stick around much longer. Compare cohorts to see whether retention is improving.
  </div>
  <div class="chart-container">{charts['course_ownership']}</div>
  <div class="explanation">
    Each dot is a course. <strong>Green dots, lower-right:</strong> "owned" courses, offered for many semesters but taught by very few people (high continuity). <strong>Red dots, upper-right:</strong> high-turnover courses, many semesters and many instructors cycling through. Dot size is total sections taught.
  </div>
  <div class="chart-container">{charts['course_ownership_top']}</div>
  <div class="explanation">
    The longest instructor-course pairings in AUS history. <strong>{most_owned_instructor}</strong> has taught <strong>{most_owned_course}</strong> for {most_owned_terms} semesters straight. These are the courses where one name has basically become the class.
  </div>

  <h3 class="chains-title" id="team-teaching">Team Teaching</h3>
  <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">The charts above credit each section to its instructor of record. AUSCrawl now captures <strong>every</strong> instructor on a section, so co-teaching finally shows up. <strong>{co_overall_pct}% of all sections have two or more instructors.</strong></p>
  <div class="chart-row">
    <div class="chart-container">{charts['co_time']}</div>
    <div class="chart-container">{charts['co_dept']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> the share of co-taught sections each semester. <strong>Right:</strong> co-teaching is mostly an <strong>engineering and lab-science</strong> thing. Biomedical (BME), Materials (MSE), and Mechatronics (MTR) lead, where capstones, design studios, and supervised labs routinely put two instructors on one section. Lecture-heavy subjects rarely bother.
  </div>
  {'<div class="chart-container">' + charts.get("co_pairs", "") + '</div>' if "co_pairs" in charts else ""}
  <div class="explanation">
    The instructor pairs who have shared the most sections. These are long-running partnerships, usually co-supervising the same lab or capstone sequence year after year.
  </div>
</section>

<!-- 5. Teaching Modality -->
<section id="modality">
  <div class="section-header">
    <div class="section-num">05 &mdash; Teaching Modality</div>
    <h2>How Is AUS Teaching?</h2>
    <p>Traditional vs. non-traditional instruction across {table_counts['semesters']} semesters. Overall non-traditional rate: {nontrad_overall_pct}%.</p>
  </div>
  <div class="chart-container">{charts['modality_over_time']}</div>
  <div class="explanation">
    How teaching methods have shifted at AUS. <strong>Traditional, in-person instruction dominates</strong> in almost every semester. The one big break was <strong>COVID-19 (2020-2021)</strong>, when non-traditional delivery spiked. After the pandemic AUS mostly went back to in-person, though a little non-traditional teaching stuck around.
  </div>
  <div class="chart-container">{charts['modality_by_subject']}</div>
  <div class="explanation">
    Subjects didn't all move online at the same rate. This ranks them by the share of sections taught in a non-traditional format. Some subjects take to online or blended formats easily; lab-heavy engineering and science courses mostly stayed in-person.
  </div>
</section>

<!-- 6. COVID-19 Impact -->
<section id="covid">
  <div class="section-header">
    <div class="section-num">06 &mdash; COVID-19 Impact</div>
    <h2>What the Data Actually Shows</h2>
    <p>AUS continued operating through COVID (Spring 2020 &ndash; Spring 2021). The shaded region marks the pandemic semesters.</p>
  </div>
  <div class="chart-container">{charts['covid_sections']}</div>
  <div class="explanation">
    Total sections barely dipped: Fall 2019 had {f19_sections}, Fall 2020 had {f20_sections} ({covid_section_change:+.1f}%). The real story is what came <em>after</em>. AUS climbed to {latest_sections} sections by {latest_term} ({growth_since:+.1f}% vs Fall 2019). Course variety (red dashed line) fell harder, down {abs(course_variety_drop):.1f}% by Spring 2021, as the university trimmed its course list but kept section counts up.
  </div>
  <div class="chart-container">{charts['covid_variety']}</div>
  <div class="explanation">
    Instructor count dropped from {f19_inst} (Fall 2019) to {f20_inst} (Fall 2020), a {abs(inst_change):.1f}% decline. With fewer unique courses too, it looks like AUS kept sections running by concentrating teaching among fewer instructors on a narrower set of courses. That reads as a coping strategy, not a collapse.
  </div>
  <div class="chart-container">{charts['covid_subjects']}</div>
  <div class="explanation">
    Each cell is a subject's section count indexed to <strong>Fall 2019 = 100</strong>. Green above 100 is growth; red below 100 is contraction. In Fall 2020, {subjects_shrank} subjects shrank by 10%+, {subjects_grew} grew by 10%+, and {subjects_stable} held steady. Language programs (ELP, ARA) and media (MCM) took the worst of it; computing (CMP, COE) and sustainability (ESM) grew. The pandemic tilted things toward technical subjects.
  </div>
  <div class="chart-container">{charts['covid_classrooms']}</div>
  <div class="explanation">
    The most lasting COVID effect: sections with no assigned physical classroom rose from {precovid_noroom}% (Fall 2019) to {latest_noroom}% ({latest_term}). Oddly, it didn't spike during COVID itself; it crept up afterward. That looks like a permanent move toward flexible or unassigned scheduling that outlasted the pandemic.
  </div>
</section>

<!-- 7. Schedule -->
<section id="schedule">
  <div class="section-header">
    <div class="section-num">07 &mdash; Schedule Patterns</div>
    <h2>When Does AUS Have Class?</h2>
    <p>AUS follows a UAE schedule: classes run Sunday through Thursday, with Saturday occasionally used.</p>
  </div>
  <div class="chart-container">{charts['schedule_heatmap']}</div>
  <div class="explanation">
    <strong>How many sections land at each day-and-time slot</strong>, across all 20 years. The busiest are <strong>Monday and Wednesday around 11:00 AM and 2:00 PM</strong>. Sunday through Thursday do the heavy lifting (the UAE work week).
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['day_patterns']}</div>
    <div class="chart-container">{charts['buildings']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> two patterns dominate, <strong>Mon/Wed (MW)</strong> and <strong>Tue/Thu/Sun (TRU)</strong>, and together they cover more than half of all sections. <strong>Right:</strong> New Academic Building 1 hosts the most sections, then the Language Building and Engineering Building Right.
  </div>
  <div class="chart-container">{charts['saturday_decline']}</div>
  <div class="explanation">
    One of the sharpest shifts in AUS scheduling. Saturday classes peaked at <strong>{sat_peak_count} sections ({sat_peak_pct}% of all sections)</strong> in <strong>{sat_peak_term}</strong>, then collapsed to near-zero by 2010. The latest semester has just {sat_recent_count} Saturday sections. In a few years AUS went from a six-day teaching week to five.
  </div>
  <div class="chart-container">{charts['day_pattern_evolution']}</div>
  <div class="explanation">
    Scheduling patterns across the full 20 years. As Saturday classes (yellow) faded out, <strong>MW</strong> and <strong>TRU</strong> took over even more. Tue/Thu (TR, no Sunday) and single-day classes hang on as smaller slices. The daily (MTWRU) pattern is mostly summer and intensive courses.
  </div>
</section>

<!-- 8. Curriculum Evolution -->
<section id="curriculum">
  <div class="section-header">
    <div class="section-num">08 &mdash; Curriculum Evolution</div>
    <h2>How the Curriculum Has Changed</h2>
    <p>{total_unique_courses:,} unique courses tracked across 20 years of offerings.</p>
  </div>
  <div class="chart-container">{charts['new_courses']}</div>
  <div class="explanation">
    How many brand-new courses showed up each year. The peak was <strong>{peak_new_year}</strong> with <strong>{peak_new_count} new courses</strong>. The early years (2005-2008) look inflated because the data starts in 2005, so courses that already existed get counted as "new" the first time they appear. Past that baseline, the rate is real curriculum growth.
  </div>
  <div class="chart-container">{charts['course_longevity']}</div>
  <div class="explanation">
    How long do courses last in the catalog? <strong>{one_sem_courses} courses ({one_sem_pct}%) ran for a single semester</strong>, usually special topics, experimental sections, or one-offs. At the other end, <strong>{veteran_courses} courses have run for 30+ semesters</strong> (15+ years). Those are the stable core of the curriculum.
  </div>
  <div class="chart-container">{charts['most_consistent']}</div>
  <div class="explanation">
    The marathon runners of the curriculum: the courses offered in the most semesters. Foundational math, English, physics, and engineering courses fill the list, because they reach the most students.
  </div>
  <div class="chart-container">{charts['courses_discontinued']}</div>
  <div class="explanation">
    Courses last offered before 2020 that had previously run for at least five semesters. That's <strong>{total_discontinued} courses that look discontinued</strong>, dropped from the active curriculum. Spikes in particular years often line up with department restructuring or program changes.
  </div>
</section>

<!-- 9. Prerequisites -->
<section id="prerequisites">
  <div class="section-header">
    <div class="section-num">09 &mdash; Prerequisite Network</div>
    <h2>The Dependency Web</h2>
    <p>{G.number_of_nodes()} courses connected by {G.number_of_edges()} prerequisite edges and {total_coreq_links:,} corequisite links.</p>
  </div>
  <div class="chart-container">{charts['prereq_connected']}</div>
  <div class="explanation">
    Red bars are how many other courses list this one as a prerequisite; blue bars are how many prerequisites it requires. Intro math, physics, and programming sit at the top, with far more courses depending on them than the other way around.
  </div>

  <h3 class="chains-title">Longest Prerequisite Chains</h3>
  <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">These are the longest sequences where each course requires the previous one. A chain of {len(longest_chains[0][1]) if longest_chains else 'N/A'} courses means a student must pass {len(longest_chains[0][1]) - 1 if longest_chains else 'N/A'} prerequisite courses before reaching the final one.</p>
  <div class="chains-grid">
  {''.join('<div class="chain"><div class="chain-header">' + str(len(path)) + ' courses &middot; ' + path[-1] + '</div>' + ''.join('<div class="chain-connector"></div><div class="chain-step"><span class="chain-step-num">' + str(i+1) + '</span>' + c + '</div>' for i, c in enumerate(path)) + '</div>' for _, path in longest_chains[:6])}
  </div>

  <div class="chart-container" style="margin-top: 2rem">{charts['coe_network']}</div>
  <div class="explanation">
    An interactive map of every <strong>Computer Engineering (COE) course</strong> and its prerequisites. Red nodes are COE courses; blue nodes are prerequisites from other departments. Hover a node to see the course name.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['prereq_complexity']}</div>
    <div class="chart-container">{charts['cross_dept']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> departments ranked by average prerequisites per course. Engineering and science carry the most complex requirement structures. <strong>Right:</strong> a matrix of which departments lean on which others for prerequisites. Most engineering departments lean hard on MTH and PHY.
  </div>

  <h3 class="chains-title" id="prereq-logic">Prerequisite Logic &amp; Complexity</h3>
  <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">The graph above counts prerequisite <em>links</em>. AUSCrawl also parses each requirement into its full <strong>boolean logic tree</strong>, the AND/OR structure students actually have to navigate. <strong>{or_share}% of courses with prerequisites offer at least one "or" alternative.</strong></p>
  <div class="chart-row">
    <div class="chart-container">{charts['prereq_shape']}</div>
    <div class="chart-container">{charts['prereq_depth']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> most prerequisites aren't a simple checklist. Plenty go beyond single-course or pure "all of these" (AND) rules, offering <strong>alternative paths</strong> (OR) or properly <strong>nested</strong> logic like "MTH 103 AND (PHY 101 OR PHY 102)". <strong>Right:</strong> how deep that nesting goes. Depth 1 is a lone course, depth 2 is a single AND/OR group, depth 3+ mixes them.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['prereq_gateways']}</div>
    <div class="chart-container">{charts['prereq_flex']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> the toughest gateways, the courses with the most <strong>mandatory</strong> prerequisites (ones that must <em>all</em> be passed, setting aside optional OR branches). <strong>{toughest.course}</strong> tops the list with {int(toughest.mandatory)}. <strong>Right:</strong> which departments build in flexibility, measured as the share of their gated courses that offer at least one alternative path instead of a rigid chain.
  </div>
  <div class="chart-container">{charts['prereq_concurrent']}</div>
  <div class="explanation">
    Some prerequisites can be taken <strong>concurrently</strong> ("may be taken together") instead of strictly beforehand, which is common for lecture/lab or theory/practice pairs. Overall, <strong>{concurrent_overall}% of prerequisite requirements</strong> allow a concurrent course, and that share has moved around over the years.
  </div>

  <h3 class="chains-title">Corequisite Analysis</h3>
  <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">Corequisites are courses that must be taken simultaneously. AUS has <strong>{total_coreq_links:,}</strong> corequisite links across the curriculum.</p>
  {'<div class="chart-container">' + charts.get("coreq_top", "") + '</div>' if "coreq_top" in charts else ""}
  <div class="explanation">
    The most common corequisite pairs are lecture-lab combos, like a physics lecture with its matching lab. Pairing them forces students to take the theory and the hands-on part in the same semester.
  </div>
  {'<div class="chart-container">' + charts.get("coreq_vs_prereq", "") + '</div>' if "coreq_vs_prereq" in charts else ""}
  <div class="explanation">
    Prerequisite vs corequisite counts by department. Most departments lean on prerequisites, but lab-intensive programs carry a lot of corequisites too.
  </div>
</section>

<!-- 10. Grades -->
<section id="grades">
  <div class="section-header">
    <div class="section-num">10 &mdash; Grade Requirements</div>
    <h2>Academic Rigor</h2>
    <p>What minimum grades do prerequisites require, and how strict are different departments?</p>
  </div>
  <div class="chart-container">{charts['grades']}</div>
  <div class="explanation">
    Each bar is how many prerequisite links require that minimum grade. <strong>C- dominates</strong>, since it's the university-wide standard pass. Next is <strong>C (no minus)</strong>, then <strong>A-</strong>, which shows up in the more competitive programs.
  </div>
  <div class="chart-container">{charts['grade_strictness']}</div>
  <div class="explanation">
    Each department's bar is the <strong>percentage breakdown</strong> of grade levels it requires for prerequisites. Departments on the left ask for the strictest grades (A or B range). For most, green (C-) is the bulk of it.
  </div>
</section>

<!-- 11. Course Attributes -->
<section id="attributes">
  <div class="section-header">
    <div class="section-num">11 &mdash; Course Attributes</div>
    <h2>Gen-Ed Tags &amp; Classifications</h2>
    <p>{total_unique_attrs} unique attributes tagging courses across the curriculum.</p>
  </div>
  <div class="chart-container">{charts['attributes_dist']}</div>
  <div class="explanation">
    Course attributes are tags that mark courses for gen-ed requirements, major electives, and special designations. <strong>"Preparatory"</strong> and <strong>"MTH Major Elective"</strong> are the most common. Science, communication, and social-science requirements fill out the rest of the top of the list, about what you'd expect from a broad gen-ed program.
  </div>
  <div class="chart-container">{charts['attributes_over_time']}</div>
  <div class="explanation">
    Attribute categories stacked over time. The rise in "Communication/English" and "Natural Sciences" tags tracks gen-ed requirements expanding. The overall jump in tagged sections mostly just follows the growth in total offerings.
  </div>

  <h3 class="chains-title" id="degree-requirements">Degree Requirement Mapping</h3>
  <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">The catalog tags each course with the degree programs it counts toward: an elective for this major, an option for that minor. Mapping those tags shows how courses get shared across programs.</p>
  <div class="chart-row">
    <div class="chart-container">{charts['program_electives']}</div>
    <div class="chart-container">{charts['reusable_courses']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> which programs give students the widest menu of elective options. <strong>Right:</strong> the most <strong>reusable</strong> courses, the single courses that count toward the most distinct programs. These are the connective tissue of the curriculum: foundational and business courses that satisfy electives across many majors and minors at once.
  </div>
</section>

<!-- 12. Catalog -->
<section id="catalog">
  <div class="section-header">
    <div class="section-num">12 &mdash; Course Catalog</div>
    <h2>Credits, Lectures, and Labs</h2>
    <p>{table_counts['catalog']:,} unique courses in the catalog.</p>
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['credit_hours']}</div>
    <div class="chart-container">{charts['lab_lecture']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> most AUS courses are worth 3 credits. Labs, independent studies, and capstones are where the exceptions live. <strong>Right:</strong> blue bars are lecture hours, red bars are lab hours, by department. Engineering and science departments rack up far more lab hours.
  </div>
  <div class="chart-container">{charts['lecture_lab']}</div>
  <div class="explanation">
    The stacked bars are raw counts of lab vs lecture sections; the green line is the lab share. That share has held fairly steady at around 15-20%.
  </div>
</section>

<!-- 13. Enrollment -->
<section id="enrollment">
  <div class="section-header">
    <div class="section-num">13 &mdash; Enrollment &amp; Access</div>
    <h2>Section Closure &amp; Enrollment Rules</h2>
    <p>What Banner reveals about demand and who is allowed to register.</p>
  </div>
  <div class="insight">
    <strong>A note on the data:</strong> AUS Banner never publishes seat counts or enrollment totals, just a binary <em>"open seats: yes/no"</em> flag captured at crawl time. For a <strong>completed</strong> term, that flag effectively records whether a section <em>closed</em> (filled up). It is not a true fill rate (enrolled ÷ capacity), which the source data can't give us. The term currently in open registration is left out of these charts, since its seats are still changing.
  </div>
  <div class="chart-container">{charts['enrollment']}</div>
  <div class="explanation">
    Each bar is a completed semester, split into sections that still <strong>had open seats (green)</strong> and those that <strong>closed with none left (red)</strong>; the line is the closure percentage. Closures have climbed steadily, from about <strong>{closure_early:.0f}%</strong> in the late 2000s to <strong>{closure_recent:.0f}%</strong> lately. That fits a growing student body packing sections more tightly.
  </div>
  <div class="chart-container">{charts['fill_rate']}</div>
  <div class="explanation">
    Subjects ranked by the share of their sections that closed (no open seats left). <strong>Higher bars point to tighter capacity and stronger demand</strong>, but keep in mind this is a binary snapshot, not a measured fill rate.
  </div>
  {'<div class="chart-container">' + charts.get("fees", "") + '</div><div class="explanation">Course fees vary by college and course type. The boxes show the spread of fee amounts, and the tiers differ from one college to the next.</div>' if "fees" in charts else ""}
  {'<div class="chart-container">' + charts.get("fee_trend", "") + '</div><div class="explanation">Fees drift upward over time, roughly tracking inflation and rising tech costs across colleges.</div>' if "fee_trend" in charts else ""}
  <div class="chart-container">{charts['restrictions_typed']}</div>
  <div class="explanation">
    Each restriction is parsed into a typed <strong>include</strong> ("must be") or <strong>exclude</strong> ("must not be") rule. <strong>{restricted_pct}% of sections carry at least one restriction</strong>, but the chart sets aside the near-universal academic-level gates (every undergraduate course is "undergraduate-only") to surface the rules that genuinely narrow who can register. Rules tied to a specific college, major, program, or field of study apply to <strong>{selective_section_pct}% of all sections</strong>.
  </div>
  <div class="chart-container">{charts['restricted_courses']}</div>
  <div class="explanation">
    {selective_restricted} courses carry a college-, major-, or program-specific rule at some point. The most-restricted ones are here: mostly senior capstones, clinical courses, and cohort-based graduate seminars held for students already in the program.
  </div>
</section>

<!-- 14. Browse Data -->
<section id="browse">
  <div class="section-header">
    <div class="section-num">14 &mdash; Browse &amp; Explore</div>
    <h2>Explore the Dataset</h2>
    <p>Search any course for its full profile and prerequisite roadmap, look up instructors, browse degree programs, or dig through the raw tables. Everything is cross-linked, so one click jumps to the next.</p>
  </div>

  <div class="tabs" role="tablist" aria-label="Browse and explore views">
    <div class="tab active" role="tab" id="tabbtn-course-explorer" aria-controls="tab-course-explorer" aria-selected="true" tabindex="0" onclick="switchTab('course-explorer')">Course Explorer</div>
    <div class="tab" role="tab" id="tabbtn-inst-explorer" aria-controls="tab-inst-explorer" aria-selected="false" tabindex="-1" onclick="switchTab('inst-explorer')">Instructor Lookup</div>
    <div class="tab" role="tab" id="tabbtn-degree-explorer" aria-controls="tab-degree-explorer" aria-selected="false" tabindex="-1" onclick="switchTab('degree-explorer')">Degree Explorer</div>
    <div class="tab" role="tab" id="tabbtn-courses" aria-controls="tab-courses" aria-selected="false" tabindex="-1" onclick="switchTab('courses')">Recent Courses</div>
    <div class="tab" role="tab" id="tabbtn-catalog" aria-controls="tab-catalog" aria-selected="false" tabindex="-1" onclick="switchTab('catalog')">Course Catalog</div>
  </div>

  <div id="tab-courses" class="tab-content" role="tabpanel" aria-labelledby="tabbtn-courses" tabindex="0">
    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.75rem;">The 5,000 most recent sections. Use the Course Explorer above for full course histories.</p>
    <div class="mini-summary"><div class="cx-label">Top subjects in this view, updating as you search or filter</div><div id="courses-summary" class="mini-bars"></div></div>
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="courses-search" aria-label="Search recent course sections" placeholder="Search courses &mdash; try COE, Calculus, or an instructor name..." oninput="filterTable('courses')">
        <select id="courses-semester" onchange="filterTable('courses')">
          <option value="">All Semesters</option>
        </select>
      </div>
      <div class="table-scroll">
        <table id="courses-table">
          <thead>
            <tr>
              <th onclick="sortTable('courses', 0)">Subject</th>
              <th onclick="sortTable('courses', 1)">Number</th>
              <th onclick="sortTable('courses', 2)">Title</th>
              <th onclick="sortTable('courses', 3)">Instructor</th>
              <th onclick="sortTable('courses', 4)">Days</th>
              <th onclick="sortTable('courses', 5)">Time</th>
              <th onclick="sortTable('courses', 6)">Room</th>
              <th onclick="sortTable('courses', 7)">Semester</th>
            </tr>
          </thead>
          <tbody id="courses-body"></tbody>
        </table>
      </div>
      <div class="table-info" id="courses-info"></div>
    </div>
  </div>

  <div id="tab-catalog" class="tab-content" role="tabpanel" aria-labelledby="tabbtn-catalog" tabindex="0">
    <div class="mini-summary"><div class="cx-label">Credit hours in this view, updating as you search</div><div id="catalog-summary" class="mini-bars"></div></div>
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="catalog-search" aria-label="Search the course catalog" placeholder="Search catalog &mdash; try a subject, keyword, or department..." oninput="filterTable('catalog')">
      </div>
      <div class="table-scroll">
        <table id="catalog-table">
          <thead>
            <tr>
              <th onclick="sortTable('catalog', 0)">Subject</th>
              <th onclick="sortTable('catalog', 1)">Number</th>
              <th onclick="sortTable('catalog', 2)">Course Name</th>
              <th onclick="sortTable('catalog', 3)">Description</th>
              <th onclick="sortTable('catalog', 4)">Credits</th>
              <th onclick="sortTable('catalog', 5)">Lecture</th>
              <th onclick="sortTable('catalog', 6)">Lab</th>
              <th onclick="sortTable('catalog', 7)">Department</th>
            </tr>
          </thead>
          <tbody id="catalog-body"></tbody>
        </table>
      </div>
      <div class="table-info" id="catalog-info"></div>
    </div>
  </div>

  <div id="tab-course-explorer" class="tab-content active" role="tabpanel" aria-labelledby="tabbtn-course-explorer" tabindex="0">
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="dep-search" aria-label="Search for a course to explore" role="combobox" aria-expanded="false" aria-controls="dep-suggestions" aria-autocomplete="list" aria-haspopup="listbox" placeholder="Type a course code &mdash; e.g. COE 420, MTH 104, PHY 101..." oninput="searchDeps(this.value)" onkeydown="comboNav(event, 'dep-suggestions', showDeps)" autocomplete="off">
      </div>
      <div id="dep-suggestions" role="listbox" aria-label="Course suggestions" style="display:none; background: var(--bg-card); border: 1px solid var(--border); border-top: 0; border-radius: 0 0 var(--radius-sm) var(--radius-sm); max-height: 220px; overflow-y: auto;"></div>
      <div id="dep-result" style="padding: 1.5rem;">
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.75rem;">Search any course for its full profile &mdash; credits, 20-year offering history, usual times, instructors &mdash; plus its prerequisite roadmap, corequisites, and what it unlocks. Try:</p>
        <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
          <span class="example-chip" onclick="showDeps('COE 420')">COE 420</span>
          <span class="example-chip" onclick="showDeps('MTH 104')">MTH 104</span>
          <span class="example-chip" onclick="showDeps('CMP 305')">CMP 305</span>
          <span class="example-chip" onclick="showDeps('FIN 201')">FIN 201</span>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-inst-explorer" class="tab-content" role="tabpanel" aria-labelledby="tabbtn-inst-explorer" tabindex="0">
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="inst-search" aria-label="Search for an instructor" role="combobox" aria-expanded="false" aria-controls="inst-suggestions" aria-autocomplete="list" aria-haspopup="listbox" placeholder="Type an instructor name..." oninput="searchInst(this.value)" onkeydown="comboNav(event, 'inst-suggestions', showInst)" autocomplete="off">
      </div>
      <div id="inst-suggestions" role="listbox" aria-label="Instructor suggestions" style="display:none; background: var(--bg-card); border: 1px solid var(--border); border-top: 0; border-radius: 0 0 var(--radius-sm) var(--radius-sm); max-height: 220px; overflow-y: auto;"></div>
      <div id="inst-result" style="padding: 1.5rem;">
        <p style="color: var(--text-muted); font-size: 0.9rem;">Search any instructor for their complete teaching history: courses taught (click any to open it in the Course Explorer), subjects, tenure, and career span at AUS.</p>
      </div>
    </div>
  </div>

  <div id="tab-degree-explorer" class="tab-content" role="tabpanel" aria-labelledby="tabbtn-degree-explorer" tabindex="0">
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="degree-search" aria-label="Search for a degree program" role="combobox" aria-expanded="false" aria-controls="degree-suggestions" aria-autocomplete="list" aria-haspopup="listbox" placeholder="Pick or type a degree program &mdash; e.g. Economics Major, FIN Minor..." oninput="searchProgram(this.value)" onfocus="searchProgram(this.value)" onblur="hideSoon('degree-suggestions')" onkeydown="comboNav(event, 'degree-suggestions', pickProgram)" autocomplete="off">
      </div>
      <div id="degree-suggestions" role="listbox" aria-label="Degree program suggestions" style="display:none; background: var(--bg-card); border: 1px solid var(--border); border-top: 0; border-radius: 0 0 var(--radius-sm) var(--radius-sm); max-height: 260px; overflow-y: auto;"></div>
      <div id="degree-result" style="padding: 1.5rem;">
        <p style="color: var(--text-muted); font-size: 0.9rem;">Pick a major or minor to see the courses the catalog tags as counting toward it &mdash; its elective options and requirement-area courses &mdash; with one-click jump to each course's full profile.</p>
      </div>
    </div>
  </div>
</section>

</main>

<!-- Footer -->
<footer>
  <p>
    Built with data from <a href="https://github.com/DeadPackets/AUSCrawl">AUSCrawl</a>
    &mdash; {table_counts['courses']:,} sections, {len(charts)} interactive charts, {table_counts['semesters']} semesters
  </p>
  <p>
    <a href="https://github.com/DeadPackets/VisualizeAUS">GitHub</a>
    &nbsp;&middot;&nbsp; Data scraped from AUS Banner &nbsp;&middot;&nbsp; MIT License
  </p>
</footer>
<script type="application/json" id="chart-data">{chart_specs_json}</script>
<script>
// ---- Data ----
const courseData = {browse_json};
const catalogData = {catalog_json};
const depData = {dep_explorer_json};
const instData = {inst_explorer_json};
const courseTitles = {course_titles_json};
const courseProfiles = {course_profiles_json};
const prereqTrees = {prereq_trees_json};
const programMap = {program_map_json};
const courseToPrograms = {course_to_programs_json};
function courseName(code) {{ return courseTitles[code] || ''; }}
function codeWithName(code) {{
  const t = courseName(code);
  return t ? code + ' <span style="color: var(--text-muted); font-weight: 400;">' + t + '</span>' : code;
}}

// ---- Theme (dark mode) ----
function isDark() {{ return document.documentElement.getAttribute('data-theme') === 'dark'; }}
const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let activeView = null;  // ['course'|'inst'|'program', id] — re-rendered on theme flip
function toggleTheme() {{
  const dark = !isDark();
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  try {{ localStorage.setItem('viz-theme', dark ? 'dark' : 'light'); }} catch (e) {{}}
  themeAllCharts(dark);
  if (activeView) {{
    if (activeView[0] === 'course') showDeps(activeView[1]);
    else if (activeView[0] === 'inst') showInst(activeView[1]);
    else if (activeView[0] === 'program') showProgram(activeView[1]);
  }}
}}

// Re-theme the baked plotly_white dashboard charts for dark mode by patching
// backgrounds, fonts, and gridlines on each graph div (no re-render needed).
function chartTheme(dark) {{
  return dark
    ? {{fg: '#cabfb0', grid: 'rgba(232,224,212,0.10)', zero: 'rgba(232,224,212,0.20)', line: 'rgba(232,224,212,0.26)'}}
    : {{fg: '#4a4a4a', grid: 'rgba(0,0,0,0.08)', zero: 'rgba(0,0,0,0.15)', line: 'rgba(0,0,0,0.20)'}};
}}
function themeChart(gd, dark) {{
  if (typeof Plotly === 'undefined' || !gd || !gd.layout) return;
  const c = chartTheme(dark);
  const patch = {{
    'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
    'font.color': c.fg, 'legend.bgcolor': 'rgba(0,0,0,0)',
    'legend.font.color': c.fg, 'legend.bordercolor': c.line, 'title.font.color': c.fg
  }};
  Object.keys(gd.layout).forEach(k => {{
    if (/^[xy]axis(\\d+)?$/.test(k)) {{
      patch[k + '.gridcolor'] = c.grid;
      patch[k + '.zerolinecolor'] = c.zero;
      patch[k + '.linecolor'] = c.line;
      patch[k + '.tickfont.color'] = c.fg;
      patch[k + '.title.font.color'] = c.fg;
    }}
  }});
  try {{ Plotly.relayout(gd, patch); }} catch (e) {{}}
}}
function themeAllCharts(dark) {{
  document.querySelectorAll('.plotly-graph-div').forEach(gd => themeChart(gd, dark));
}}

// Populate semester dropdown
const semesters = [...new Set(courseData.map(r => r.term_name))];
const semSelect = document.getElementById('courses-semester');
semesters.forEach(s => {{
  const opt = document.createElement('option');
  opt.value = s; opt.textContent = s;
  semSelect.appendChild(opt);
}});

// Render tables
function renderTable(type, data) {{
  const body = document.getElementById(type + '-body');
  const info = document.getElementById(type + '-info');
  const rows = data.slice(0, 500);

  if (type === 'courses') {{
    body.innerHTML = rows.map(r => `<tr>
      <td>${{r.subject}}</td><td>${{r.course_number}}</td>
      <td title="${{r.title}}">${{r.title}}</td>
      <td>${{r.instructor_name || 'TBA'}}</td>
      <td>${{r.days || '-'}}</td>
      <td>${{r.start_time ? r.start_time + ' - ' + r.end_time : '-'}}</td>
      <td>${{r.classroom || 'TBA'}}</td>
      <td>${{r.term_name}}</td>
    </tr>`).join('');
  }} else {{
    body.innerHTML = rows.map(r => `<tr>
      <td>${{r.subject}}</td><td>${{r.course_number}}</td>
      <td>${{r.title || '-'}}</td>
      <td title="${{r.description || ''}}">${{(r.description || '').substring(0, 80)}}${{(r.description || '').length > 80 ? '...' : ''}}</td>
      <td>${{r.credit_hours || '-'}}</td>
      <td>${{r.lecture_hours || '-'}}</td>
      <td>${{r.lab_hours || '-'}}</td>
      <td>${{(r.department || '').replace(' Department', '')}}</td>
    </tr>`).join('');
  }}
  info.textContent = `Showing ${{rows.length}} of ${{data.length}} records`;
}}

function filterTable(type) {{
  const search = document.getElementById(type + '-search').value.toLowerCase();
  let data = type === 'courses' ? courseData : catalogData;

  if (type === 'courses') {{
    const sem = document.getElementById('courses-semester').value;
    if (sem) data = data.filter(r => r.term_name === sem);
  }}

  if (search) {{
    data = data.filter(r => Object.values(r).some(v =>
      v && String(v).toLowerCase().includes(search)));
  }}

  renderTable(type, data);
  renderTableSummary(type, data);
}}

// Lightweight, fast-redrawing CSS bars that track the current table filter.
function renderMiniBars(id, items, color) {{
  const el = document.getElementById(id);
  if (!el) return;
  if (!items.length) {{ el.innerHTML = '<span style="color: var(--text-light); font-size: 0.8rem;">No matches.</span>'; return; }}
  const max = Math.max.apply(null, items.map(i => i[1])) || 1;
  el.innerHTML = items.map(i =>
    `<div class="mb-row"><span class="mb-label" title="${{i[0]}}">${{i[0]}}</span><span class="mb-track"><span class="mb-fill" style="width: ${{(i[1] / max * 100).toFixed(1)}}%; background: ${{color || '#C4972F'}};"></span></span><span class="mb-val">${{i[1].toLocaleString()}}</span></div>`
  ).join('');
}}

function renderTableSummary(type, data) {{
  if (type === 'courses') {{
    const c = {{}};
    data.forEach(r => {{ c[r.subject] = (c[r.subject] || 0) + 1; }});
    const top = Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, 8);
    renderMiniBars('courses-summary', top, '#C4972F');
  }} else {{
    const c = {{}};
    data.forEach(r => {{
      const cr = (r.credit_hours == null || r.credit_hours === '') ? '?' : (Math.round(r.credit_hours) + ' cr');
      c[cr] = (c[cr] || 0) + 1;
    }});
    const items = Object.entries(c).sort((a, b) => {{
      const pa = a[0] === '?' ? Infinity : parseFloat(a[0]);
      const pb = b[0] === '?' ? Infinity : parseFloat(b[0]);
      return pa - pb;
    }});
    renderMiniBars('catalog-summary', items, '#2471a3');
  }}
}}

let sortState = {{}};
function sortTable(type, col) {{
  const key = type + col;
  sortState[key] = !(sortState[key] || false);
  const asc = sortState[key];

  let data = type === 'courses' ? [...courseData] : [...catalogData];
  const keys = type === 'courses'
    ? ['subject','course_number','title','instructor_name','days','start_time','classroom','term_name']
    : ['subject','course_number','title','description','credit_hours','lecture_hours','lab_hours','department'];

  data.sort((a, b) => {{
    const va = a[keys[col]] || '';
    const vb = b[keys[col]] || '';
    return asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  }});

  if (type === 'courses') courseData.splice(0, courseData.length, ...data);
  else catalogData.splice(0, catalogData.length, ...data);
  filterTable(type);
}}

function switchTab(tab) {{
  document.querySelectorAll('.tab').forEach(t => {{
    const on = t.getAttribute('aria-controls') === 'tab-' + tab;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', on ? 'true' : 'false');
    t.tabIndex = on ? 0 : -1;            // roving tabindex for the tablist
  }});
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  const el = document.getElementById('tab-' + tab);
  if (el) el.classList.add('active');
}}
function gotoCourse(code) {{ switchTab('course-explorer'); showDeps(code); }}
function gotoInst(name) {{ switchTab('inst-explorer'); showInst(name); }}
function gotoProgram(name) {{ switchTab('degree-explorer'); document.getElementById('degree-search').value = name; showProgram(name); }}

// ---- Plotly mini-chart helpers (on demand, theme- and motion-aware) -------
const PCFG = {{responsive: true, displayModeBar: false}};
const PANIM = {{transition: {{duration: 650, easing: 'cubic-out'}}, frame: {{duration: 650}}}};
function gridColor() {{ return isDark() ? '#3a342c' : '#ece8e0'; }}
function fgColor() {{ return isDark() ? '#c8c0b2' : '#4a4a4a'; }}
function plLayout(extra) {{
  const grid = gridColor();
  return Object.assign({{
    height: 210, margin: {{l: 38, r: 12, t: 8, b: 30}},
    font: {{family: 'Montserrat, sans-serif', size: 11, color: fgColor()}},
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    bargap: 0.22, showlegend: false,
    xaxis: {{gridcolor: grid, zeroline: false, automargin: true, fixedrange: true}},
    yaxis: {{gridcolor: grid, zeroline: false, automargin: true, fixedrange: true}}
  }}, extra || {{}});
}}
// Vertical bars (categories x, values y) that grow from zero. The axis range
// is pinned to the data max so the grow-in animation doesn't clip.
function colChart(id, x, y, color, hov, extra) {{
  const el = document.getElementById(id);
  if (!el) return;
  const mx = (Math.max.apply(null, y) || 1) * 1.12;
  const lay = plLayout(Object.assign(
    {{yaxis: {{gridcolor: gridColor(), zeroline: false, automargin: true, fixedrange: true, range: [0, mx]}}}},
    extra || {{}}));
  if (prefersReduced) {{ Plotly.newPlot(id, [{{type: 'bar', x: x, y: y, marker: {{color: color, line: {{width: 0}}}}, hovertemplate: hov}}], lay, PCFG); return; }}
  Plotly.newPlot(id, [{{type: 'bar', x: x, y: y.map(() => 0), marker: {{color: color, line: {{width: 0}}}}, hovertemplate: hov}}], lay, PCFG)
    .then(() => Plotly.animate(id, {{data: [{{y: y}}]}}, PANIM));
}}
// Horizontal bars (categories y, values x) that grow from zero.
function rowChart(id, cats, vals, color, hov, h) {{
  const el = document.getElementById(id);
  if (!el) return;
  const mx = (Math.max.apply(null, vals) || 1) * 1.12;
  const lay = plLayout({{height: h || 210, margin: {{l: 8, r: 14, t: 8, b: 26}},
    xaxis: {{gridcolor: gridColor(), zeroline: false, automargin: true, fixedrange: true, range: [0, mx]}},
    yaxis: {{automargin: true, fixedrange: true, ticksuffix: '  '}}}});
  if (prefersReduced) {{ Plotly.newPlot(id, [{{type: 'bar', orientation: 'h', y: cats, x: vals, marker: {{color: color, line: {{width: 0}}}}, hovertemplate: hov}}], lay, PCFG); return; }}
  Plotly.newPlot(id, [{{type: 'bar', orientation: 'h', y: cats, x: vals.map(() => 0), marker: {{color: color, line: {{width: 0}}}}, hovertemplate: hov}}], lay, PCFG)
    .then(() => Plotly.animate(id, {{data: [{{x: vals}}]}}, PANIM));
}}

// ---- Course Explorer ----
function statBox(v, label) {{
  return `<div><div style="font-family: var(--font-mono); font-size: 1.5rem; font-weight: 600; color: var(--text); line-height: 1;">${{v}}</div><div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem;">${{label}}</div></div>`;
}}

// Shared keyboard support for the autocomplete comboboxes: ArrowUp/Down move
// the active suggestion (tracked via aria-selected + aria-activedescendant),
// Enter chooses it, Escape closes the list. Typing keys fall through.
function comboNav(e, boxId, selectFn) {{
  const box = document.getElementById(boxId);
  if (!box || box.style.display === 'none') return;
  const opts = [...box.querySelectorAll('[role="option"]')];
  if (!opts.length) return;
  const input = e.currentTarget;
  const idx = opts.findIndex(o => o.getAttribute('aria-selected') === 'true');
  const setActive = (i) => {{
    opts.forEach((o, j) => o.setAttribute('aria-selected', j === i ? 'true' : 'false'));
    if (i >= 0) {{ opts[i].scrollIntoView({{ block: 'nearest' }}); input.setAttribute('aria-activedescendant', opts[i].id); }}
    else input.removeAttribute('aria-activedescendant');
  }};
  if (e.key === 'ArrowDown') {{ e.preventDefault(); setActive(idx < opts.length - 1 ? idx + 1 : 0); }}
  else if (e.key === 'ArrowUp') {{ e.preventDefault(); setActive(idx > 0 ? idx - 1 : opts.length - 1); }}
  else if (e.key === 'Enter' && idx >= 0) {{ e.preventDefault(); input.setAttribute('aria-expanded', 'false'); selectFn(opts[idx].getAttribute('data-val')); }}
  else if (e.key === 'Escape') {{ box.style.display = 'none'; input.setAttribute('aria-expanded', 'false'); input.removeAttribute('aria-activedescendant'); }}
}}

function searchDeps(query) {{
  const q = query.toUpperCase().trim();
  const sugBox = document.getElementById('dep-suggestions');
  const input = document.getElementById('dep-search');
  if (q.length < 2) {{ sugBox.style.display = 'none'; input.setAttribute('aria-expanded', 'false'); return; }}
  const matches = Object.keys(courseProfiles).filter(k => k.includes(q)).sort().slice(0, 14);
  if (matches.length === 0) {{ sugBox.style.display = 'none'; input.setAttribute('aria-expanded', 'false'); return; }}
  sugBox.style.display = 'block';
  input.setAttribute('aria-expanded', 'true');
  input.removeAttribute('aria-activedescendant');
  sugBox.innerHTML = matches.map((m, i) =>
    `<div class="sugg-opt" role="option" id="dep-suggestions-opt-${{i}}" data-val="${{m}}" aria-selected="false"
          style="padding: 0.5rem 1rem; cursor: pointer; font-size: 0.85rem; border-bottom: 1px solid var(--border-light);"
          onclick="showDeps('${{m}}')"><span style="font-family: var(--font-mono); font-weight: 500;">${{m}}</span> <span style="color: var(--text-muted);">${{courseName(m)}}</span></div>`).join('');
}}

// Recursive AND/OR prerequisite roadmap. Auto-expands one level into each
// direct prerequisite; deeper levels show a "&rsaquo;" hint (click to drill in).
function renderReqTree(node, visited, depth) {{
  const MAXD = 1;
  if (node.o) {{
    const kids = node.x.map(k => renderReqTree(k, visited, depth)).join('');
    return `<div class="cx-node"><span class="cx-op ${{node.o}}">${{node.o === 'and' ? 'all of' : 'one of'}}</span><div class="cx-kids">${{kids}}</div></div>`;
  }}
  const code = node.c;
  const sub = prereqTrees[code];
  const expand = sub && depth < MAXD && !visited.has(code);
  let leaf = `<span class="cx-leaf" onclick="showDeps('${{code}}')"><span class="mono">${{code}}</span>` +
    (node.g ? ` <span style="color: var(--accent);">(min ${{node.g}})</span>` : '') +
    (node.k ? ` <span style="color: #16a085;">&#8635; concurrent</span>` : '') +
    ` <span class="nm">${{courseName(code)}}</span>` +
    (sub && !expand ? ' <span class="exp" title="has its own prerequisites">&rsaquo;</span>' : '') + `</span>`;
  let html = `<div class="cx-node">${{leaf}}`;
  if (expand) {{
    const v2 = new Set(visited); v2.add(code);
    html += `<div class="cx-kids">${{renderReqTree(sub, v2, depth + 1)}}</div>`;
  }}
  return html + `</div>`;
}}

function showDeps(course) {{
  activeView = ['course', course];
  document.getElementById('dep-suggestions').style.display = 'none';
  document.getElementById('dep-search').setAttribute('aria-expanded', 'false');
  document.getElementById('dep-search').value = course;
  const result = document.getElementById('dep-result');
  const p = courseProfiles[course];
  const dep = depData[course];
  const tree = prereqTrees[course];
  if (!p && !dep) {{ result.innerHTML = '<p style="color: var(--text-muted);">Course not found.</p>'; return; }}

  const nm = courseName(course);
  let html = `<h3 style="font-family: var(--font-display); margin-bottom: 0.75rem; color: var(--accent);">${{course}}${{nm ? ' <span style="color: var(--text-secondary); font-weight: 500; font-size: 1.05rem;">' + nm + '</span>' : ''}}</h3>`;

  if (p) {{
    html += '<div style="display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;">';
    if (p.cr != null) html += statBox(p.cr % 1 ? p.cr.toFixed(1) : p.cr, 'Credits');
    html += statBox(p.ns.toLocaleString(), 'Sections');
    html += statBox(p.nt, 'Semesters');
    if (p.ni) html += statBox(p.ni, 'Instructors');
    html += '</div>';
    if (p.yc && p.yc.length) html += `<div class="cx-section"><div class="cx-label">Sections offered per year &middot; ${{p.ft}} &ndash; ${{p.lt}}</div><div id="cx-sections" style="min-height: 200px;"></div></div>`;
    if (p.dy || p.tm) html += `<div class="cx-section"><div class="cx-label">Usually meets</div><div style="font-size: 0.9rem; color: var(--text-secondary);">${{p.dy ? '<strong>' + p.dy + '</strong>' : ''}}${{p.tm ? ' &middot; ' + p.tm : ''}}</div></div>`;
    if (p.ins && p.ins.length) {{
      html += `<div class="cx-section"><div class="cx-label">Taught by${{p.ni ? ' (' + p.ni + ' total)' : ''}}</div><div class="cx-chips">` +
        p.ins.map(i => `<span class="cx-chip" onclick="gotoInst('${{i[0].replace(/'/g, "\\\\'")}}')"><span style="color: var(--text);">${{i[0]}}</span> <span style="color: var(--text-muted);">&middot; ${{i[1]}}</span></span>`).join('') + `</div></div>`;
    }}
  }}

  const ctp = courseToPrograms[course];
  if (ctp) {{
    html += `<div class="cx-section"><div class="cx-label">Counts toward ${{ctp[0]}} degree program${{ctp[0] !== 1 ? 's' : ''}}</div><div class="cx-chips">` +
      ctp[1].map(pn => `<span class="cx-chip prog" onclick="gotoProgram('${{pn.replace(/'/g, "\\\\'")}}')">${{pn}}</span>`).join('') +
      (ctp[0] > ctp[1].length ? `<span style="color: var(--text-muted); font-size: 0.82rem; align-self: center;">+${{ctp[0] - ctp[1].length}} more</span>` : '') + `</div></div>`;
  }}

  if (tree) {{
    html += `<div class="cx-section" style="margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px solid var(--border);"><div class="cx-label">Prerequisite Roadmap</div><div class="cx-tree">${{renderReqTree(tree, new Set([course]), 0)}}</div></div>`;
  }} else if (dep && dep.p.length) {{
    html += `<div class="cx-section" style="margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px solid var(--border);"><div class="cx-label">Prerequisites (${{dep.p.length}})</div><div class="cx-chips">` +
      dep.p.map(pp => `<span class="cx-chip pre" onclick="showDeps('${{pp.c}}')"><span class="mono">${{pp.c}}</span>${{pp.g ? ' <span style="color: var(--accent);">(min ' + pp.g + ')</span>' : ''}} <span class="nm">${{courseName(pp.c)}}</span></span>`).join('') + `</div></div>`;
  }}

  const coreqs = (dep && dep.q) || [];
  if (coreqs.length) {{
    html += `<div class="cx-section"><div class="cx-label">Corequisites (${{coreqs.length}})</div><div class="cx-chips">` +
      coreqs.map(c => `<span class="cx-chip co" onclick="showDeps('${{c}}')"><span class="mono">${{c}}</span> <span class="nm">${{courseName(c)}}</span></span>`).join('') + `</div></div>`;
  }}

  const nexts = (dep && dep.n) || [];
  if (nexts.length) {{
    html += `<div class="cx-section"><div class="cx-label">Unlocks &mdash; is a prerequisite for (${{nexts.length}})</div><div class="cx-chips">` +
      nexts.map(c => `<span class="cx-chip next" onclick="showDeps('${{c}}')"><span class="mono">${{c}}</span> <span class="nm">${{courseName(c)}}</span></span>`).join('') + `</div></div>`;
  }}

  result.innerHTML = html;

  // Animated sections-per-year chart (fill dormant years with zero).
  if (p && p.yc && p.yc.length) {{
    const cnt = {{}};
    p.yc.forEach(d => cnt[d[0]] = d[1]);
    const lo = Math.min.apply(null, p.yc.map(d => d[0])), hi = Math.max.apply(null, p.yc.map(d => d[0]));
    const xs = [], vs = [];
    for (let y = lo; y <= hi; y++) {{ xs.push(String(y)); vs.push(cnt[y] || 0); }}
    colChart('cx-sections', xs, vs, '#C4972F', '%{{x}}: %{{y}} sections<extra></extra>', {{height: 195}});
  }}
}}

// ---- Instructor Explorer ----
function searchInst(query) {{
  const q = query.toLowerCase().trim();
  const sugBox = document.getElementById('inst-suggestions');
  const input = document.getElementById('inst-search');
  if (q.length < 2) {{ sugBox.style.display = 'none'; input.setAttribute('aria-expanded', 'false'); return; }}
  const matches = Object.keys(instData).filter(k => k.toLowerCase().includes(q)).sort().slice(0, 14);
  if (matches.length === 0) {{ sugBox.style.display = 'none'; input.setAttribute('aria-expanded', 'false'); return; }}
  sugBox.style.display = 'block';
  input.setAttribute('aria-expanded', 'true');
  input.removeAttribute('aria-activedescendant');
  sugBox.innerHTML = matches.map((m, i) =>
    `<div class="sugg-opt" role="option" id="inst-suggestions-opt-${{i}}" data-val="${{m.replace(/"/g, '&quot;')}}" aria-selected="false"
          style="padding: 0.5rem 1rem; cursor: pointer; font-size: 0.85rem; border-bottom: 1px solid var(--border-light);"
          onclick="showInst('${{m.replace(/'/g, "\\\\'")}}')"><strong>${{m}}</strong></div>`).join('');
}}

function showInst(name) {{
  activeView = ['inst', name];
  document.getElementById('inst-suggestions').style.display = 'none';
  document.getElementById('inst-search').setAttribute('aria-expanded', 'false');
  document.getElementById('inst-search').value = name;
  const data = instData[name];
  const result = document.getElementById('inst-result');
  if (!data) {{ result.innerHTML = '<p style="color: var(--text-muted);">Instructor not found.</p>'; return; }}

  let html = `<h3 style="font-family: var(--font-display); margin-bottom: 0.75rem; color: var(--accent);">${{name}}</h3>`;
  html += '<div style="display: flex; gap: 2rem; margin-bottom: 1rem; flex-wrap: wrap;">';
  html += statBox(data.t.toLocaleString(), 'Sections') + statBox(data.n, 'Semesters') + statBox(data.s.length, 'Subjects');
  html += '</div>';
  html += `<p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;"><strong>Active:</strong> ${{data.f}} &mdash; ${{data.l}}</p>`;
  if (data.a && data.a.length > 1) html += `<div class="cx-section"><div class="cx-label">Sections taught per year</div><div id="inst-timeline" style="min-height: 200px;"></div></div>`;
  if (data.sm && data.sm.length > 1) html += `<div class="cx-section"><div class="cx-label">Teaching by subject</div><div id="inst-subjects" style="min-height: 180px;"></div></div>`;
  html += '<div class="cx-section"><div class="cx-label">Subjects Taught</div><div class="cx-chips">' + data.s.map(s =>
    `<span style="background: var(--bg-warm); border: 1px solid var(--border); padding: 0.2rem 0.6rem; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.82rem;">${{s}}</span>`
  ).join('') + '</div></div>';
  html += '<div class="cx-section"><div class="cx-label">Top Courses &mdash; click to open in the Course Explorer</div><div class="cx-chips">' +
    data.c.map(c => `<span class="cx-chip" onclick="gotoCourse('${{c[0]}} ${{c[1]}}')"><span class="mono">${{c[0]}} ${{c[1]}}</span> <span class="nm">${{c[2]}}</span> <span style="color: var(--text-muted);">&middot; ${{c[3]}}&times;</span></span>`).join('') + '</div></div>';
  result.innerHTML = html;

  if (data.a && data.a.length > 1) {{
    const cnt = {{}};
    data.a.forEach(d => cnt[d[0]] = d[1]);
    const lo = Math.min.apply(null, data.a.map(d => d[0])), hi = Math.max.apply(null, data.a.map(d => d[0]));
    const xs = [], vs = [];
    for (let y = lo; y <= hi; y++) {{ xs.push(String(y)); vs.push(cnt[y] || 0); }}
    colChart('inst-timeline', xs, vs, '#9e3223', '%{{x}}: %{{y}} sections<extra></extra>', {{height: 195}});
  }}
  if (data.sm && data.sm.length > 1) {{
    const cats = data.sm.map(d => d[0]).reverse(), vals = data.sm.map(d => d[1]).reverse();
    rowChart('inst-subjects', cats, vals, '#2471a3', '%{{y}}: %{{x}} sections<extra></extra>', Math.max(150, cats.length * 26));
  }}
}}

// ---- Degree Explorer ----
function hideSoon(id) {{ setTimeout(() => {{ const b = document.getElementById(id); if (b) b.style.display = 'none'; }}, 150); }}

function searchProgram(query) {{
  const q = query.toLowerCase().trim();
  const box = document.getElementById('degree-suggestions');
  const input = document.getElementById('degree-search');
  const keys = Object.keys(programMap).sort();
  const matches = (q ? keys.filter(k => k.toLowerCase().includes(q)) : keys).slice(0, 60);
  if (!matches.length) {{ box.style.display = 'none'; input.setAttribute('aria-expanded', 'false'); return; }}
  box.style.display = 'block';
  input.setAttribute('aria-expanded', 'true');
  input.removeAttribute('aria-activedescendant');
  box.innerHTML = matches.map((m, i) =>
    `<div class="sugg-opt" role="option" id="degree-suggestions-opt-${{i}}" data-val="${{m.replace(/"/g, '&quot;')}}" aria-selected="false"
          style="padding: 0.5rem 1rem; cursor: pointer; font-size: 0.85rem; border-bottom: 1px solid var(--border-light);"
          onmousedown="pickProgram('${{m.replace(/'/g, "\\\\'")}}')">${{m}} <span style="color: var(--text-muted);">&middot; ${{programMap[m].length}} courses</span></div>`).join('');
}}
function pickProgram(name) {{
  document.getElementById('degree-search').value = name;
  document.getElementById('degree-suggestions').style.display = 'none';
  document.getElementById('degree-search').setAttribute('aria-expanded', 'false');
  showProgram(name);
}}

// Banner's course_attributes only ever express elective options and
// requirement-area tags — never a program's required core sequence (that lives
// in the degree plan). So label the buckets honestly rather than calling
// everything an "elective".
const ROLE_LABELS = {{
  Required: 'Required courses', Core: 'Core courses',
  Elective: 'Elective options', Other: 'Requirement-area &amp; other courses'
}};
function showProgram(name) {{
  const result = document.getElementById('degree-result');
  const courses = programMap[name];
  if (!courses) {{ return; }}
  activeView = ['program', name];
  const groups = {{ Required: [], Core: [], Elective: [], Other: [] }};
  const subjCount = {{}};
  courses.forEach(pair => {{
    (groups[pair[1]] || groups.Other).push(pair[0]);
    const subj = pair[0].split(' ')[0];
    subjCount[subj] = (subjCount[subj] || 0) + 1;
  }});
  let html = `<h3 style="font-family: var(--font-display); margin-bottom: 0.75rem; color: var(--accent);">${{name}} <span style="color: var(--text-muted); font-weight: 500; font-size: 0.9rem;">(${{courses.length}} courses)</span></h3>`;
  html += `<div class="cx-note">These are the courses AUS's catalog tags as counting toward <strong>${{name}}</strong> &mdash; its elective options and requirement-area courses. A program's required core sequence is defined in the degree plan, not as course attributes, so it isn't fully captured here.</div>`;
  html += `<div class="cx-section"><div class="cx-label">Course pool by subject (top 15)</div><div id="degree-subjects" style="min-height: 200px;"></div></div>`;
  ['Required', 'Core', 'Elective', 'Other'].forEach(role => {{
    const list = groups[role];
    if (!list.length) return;
    const chips = list.sort().map(code => `<span class="cx-chip" onclick="gotoCourse('${{code}}')"><span class="mono">${{code}}</span> <span class="nm">${{courseName(code)}}</span></span>`).join('');
    const label = `<div class="cx-label">${{ROLE_LABELS[role] || role}} (${{list.length}})</div>`;
    if (list.length > 24) {{
      html += `<div class="cx-section">${{label}}<div class="cx-scrollbox"><div class="cx-chips scrolly">${{chips}}</div><div class="cx-fade"></div><div class="cx-more-hint">scroll for more &darr;</div></div></div>`;
    }} else {{
      html += `<div class="cx-section">${{label}}<div class="cx-chips">${{chips}}</div></div>`;
    }}
  }});
  result.innerHTML = html;
  wireScrollboxes(result);

  const top = Object.entries(subjCount).sort((a, b) => b[1] - a[1]).slice(0, 15);
  if (top.length) {{
    const cats = top.map(d => d[0]).reverse(), vals = top.map(d => d[1]).reverse();
    rowChart('degree-subjects', cats, vals, '#27ae60', '%{{y}}: %{{x}} courses<extra></extra>', Math.max(160, cats.length * 24));
  }}
}}

// Toggle the "more below" fade/hint on any scrollable chip pool inside `root`.
function wireScrollboxes(root) {{
  (root || document).querySelectorAll('.cx-scrollbox').forEach(box => {{
    const sc = box.querySelector('.scrolly');
    if (!sc) return;
    const upd = () => box.classList.toggle('more', sc.scrollTop + sc.clientHeight < sc.scrollHeight - 4);
    sc.addEventListener('scroll', upd, {{ passive: true }});
    requestAnimationFrame(upd);
  }});
}}

// Initial render (filterTable also paints the live summary mini-charts)
filterTable('courses');
filterTable('catalog');

// Active nav on scroll — highlight the current section's link, keep it scrolled
// into view within the horizontal nav, and expose it to assistive tech.
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('nav a[href^="#"]:not(.nav-brand)');
let activeNavId = '';
function syncActiveNav() {{
  let current = '';
  sections.forEach(s => {{ if (window.scrollY >= s.offsetTop - 200) current = s.id; }});
  if (current === activeNavId) return;   // only react when the section changes
  activeNavId = current;
  let activeLink = null;
  navLinks.forEach(a => {{
    const on = a.getAttribute('href') === '#' + current;
    a.classList.toggle('active', on);
    if (on) {{ a.setAttribute('aria-current', 'true'); activeLink = a; }}
    else a.removeAttribute('aria-current');
  }});
  // Center the active link inside the scrollable nav (horizontal only, so the
  // page's vertical scroll is untouched). Honors reduced-motion.
  const inner = document.getElementById('nav-inner');
  if (inner) {{
    if (activeLink) {{
      const ir = inner.getBoundingClientRect(), lr = activeLink.getBoundingClientRect();
      const delta = (lr.left + lr.width / 2) - (ir.left + ir.width / 2);
      if (Math.abs(delta) > 4) inner.scrollBy({{ left: delta, behavior: prefersReduced ? 'auto' : 'smooth' }});
    }} else {{
      inner.scrollTo({{ left: 0, behavior: prefersReduced ? 'auto' : 'smooth' }});  // back at the hero
    }}
  }}
}}
window.addEventListener('scroll', syncActiveNav, {{ passive: true }});
syncActiveNav();

// Nav horizontal-scroll affordance: fade the edge(s) that have more links.
(function () {{
  const scroller = document.getElementById('nav-inner');
  const wrap = document.getElementById('nav-scroll');
  if (!scroller || !wrap) return;
  const upd = () => {{
    const max = scroller.scrollWidth - scroller.clientWidth;
    wrap.classList.toggle('more-left', scroller.scrollLeft > 2);
    wrap.classList.toggle('more-right', scroller.scrollLeft < max - 2);
  }};
  scroller.addEventListener('scroll', upd, {{ passive: true }});
  window.addEventListener('resize', upd, {{ passive: true }});
  upd();
}})();

// Keyboard support for the Browse tablist (WAI-ARIA pattern): arrow keys /
// Home / End move focus and activate; Enter / Space activate.
(function () {{
  const tablist = document.querySelector('.tabs[role="tablist"]');
  if (!tablist) return;
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  tablist.addEventListener('keydown', (e) => {{
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    let n = -1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') n = (i + 1) % tabs.length;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') n = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') n = 0;
    else if (e.key === 'End') n = tabs.length - 1;
    else if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); tabs[i].click(); return; }}
    else return;
    e.preventDefault();
    tabs[n].focus();
    tabs[n].click();
  }});
}})();

// Lazy chart rendering: only draw a chart when it scrolls near the viewport,
// instead of laying out all 62 on load. Plotly is loaded with `defer`, so we
// wait for DOMContentLoaded (deferred scripts run before it) to be sure it's
// ready. Each chart re-themes itself for dark mode at render time.
function setupLazyCharts() {{
  let specs;
  try {{ specs = JSON.parse(document.getElementById('chart-data').textContent); }}
  catch (e) {{ return; }}
  const draw = (el) => {{
    if (el.dataset.rendered) return;
    const spec = specs[el.getAttribute('data-cid')];
    if (!spec || typeof Plotly === 'undefined') return;
    el.dataset.rendered = '1';
    Plotly.newPlot(el, spec.data, spec.layout, {{responsive: true}}).then(() => {{
      el.style.minHeight = '';
      if (isDark()) themeChart(el, true);
    }});
  }};
  // Generous lead margin: render charts ~600px before they enter view so the
  // (one-time) Plotly init cost is paid off-screen and no empty box flashes.
  const io = new IntersectionObserver((entries, obs) => {{
    entries.forEach(e => {{ if (e.isIntersecting) {{ obs.unobserve(e.target); draw(e.target); }} }});
  }}, {{ rootMargin: '600px 0px' }});
  document.querySelectorAll('.lazy-chart').forEach(el => io.observe(el));
}}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setupLazyCharts);
else setupLazyCharts();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "index.html").write_text(html)
print(f"Built site: {OUT_DIR / 'index.html'}")
print(f"  Charts: {len(charts)}")
print(f"  File size: {len(html) / 1024 / 1024:.1f} MB")
print(f"  Course table: {len(df_browse_recent)} rows")
print(f"  Catalog table: {len(df_cat_browse)} rows")
