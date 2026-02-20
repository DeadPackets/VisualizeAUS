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

def chart_html(fig, chart_id):
    """Return a Plotly figure as an embeddable HTML div."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=f"chart-{chart_id}", config={"responsive": True})


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

# ===== 8. Enrollment =====
df_seats = pd.read_sql_query("""
    SELECT s.term_name, s.term_id,
           SUM(CASE WHEN c.seats_available = 1 THEN 1 ELSE 0 END) as available,
           SUM(CASE WHEN c.seats_available = 0 THEN 1 ELSE 0 END) as full_sections,
           COUNT(*) as total,
           ROUND(100.0 * SUM(CASE WHEN c.seats_available = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as full_pct
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE s.term_name LIKE 'Fall%' OR s.term_name LIKE 'Spring%'
    GROUP BY c.term_id ORDER BY c.term_id
""", conn)
fig = go.Figure()
fig.add_trace(go.Bar(x=df_seats["term_name"], y=df_seats["available"], name="Available", marker_color="#2ecc71"))
fig.add_trace(go.Bar(x=df_seats["term_name"], y=df_seats["full_sections"], name="Full", marker_color="#e74c3c"))
fig.update_layout(template=TEMPLATE, title="Section Availability Per Semester",
                  barmode="stack", height=450, xaxis=dict(tickangle=-45, dtick=4),
                  yaxis_title="Sections",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
charts["enrollment"] = chart_html(fig, "enrollment")

# ===== 8b. Subject fill rate =====
df_subj_full = pd.read_sql_query("""
    SELECT subject,
           SUM(CASE WHEN seats_available = 0 THEN 1 ELSE 0 END) as full_count,
           COUNT(*) as total,
           ROUND(100.0 * SUM(CASE WHEN seats_available = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as full_pct
    FROM courses GROUP BY subject HAVING total >= 50 ORDER BY full_pct DESC
""", conn)
fig = px.bar(df_subj_full.head(25), x="subject", y="full_pct",
             color="full_pct", color_continuous_scale="RdYlGn_r",
             hover_data=["full_count", "total"],
             title="Subject Fill Rate: % of Sections That Were Full",
             labels={"full_pct": "% Full", "subject": "Subject"})
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

# ===== NEW: Enrollment Restrictions =====
df_restrictions = pd.read_sql_query("""
    SELECT sd.restrictions, COUNT(*) as sections
    FROM section_details sd
    WHERE sd.restrictions != ''
    GROUP BY sd.restrictions
""", conn)

def categorize_restriction(r):
    r_lower = str(r).lower()
    if 'level' in r_lower and 'undergraduate' in r_lower and 'graduate' not in r_lower: return 'Undergraduate Only'
    if 'level' in r_lower and 'graduate' in r_lower and 'undergraduate' not in r_lower: return 'Graduate Only'
    if 'level' in r_lower: return 'Level Restriction (Mixed)'
    if 'major' in r_lower or 'program' in r_lower: return 'Major/Program'
    if 'college' in r_lower or 'school' in r_lower: return 'College/School'
    if 'class' in r_lower or 'standing' in r_lower: return 'Class Standing'
    if 'department' in r_lower: return 'Department'
    return 'Other'

restriction_cats = Counter()
for _, row in df_restrictions.iterrows():
    cat = categorize_restriction(row['restrictions'])
    restriction_cats[cat] += row['sections']

df_restrict = pd.DataFrame(restriction_cats.most_common(), columns=['category', 'sections'])
df_restrict['pct'] = (df_restrict['sections'] / df_restrict['sections'].sum() * 100).round(1)

fig = px.pie(df_restrict, values='sections', names='category',
             title="Types of Enrollment Restrictions",
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(template=TEMPLATE, height=450)
charts['restriction_types'] = chart_html(fig, 'restriction_types')

total_restricted = int(df_restrictions['sections'].sum())
restricted_pct = round(total_restricted / table_counts['section_details'] * 100, 1)

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
    SELECT subject, course_number, description, credit_hours, lecture_hours,
           lab_hours, department FROM catalog ORDER BY subject, course_number
""", conn2)
catalog_json = df_cat_browse.to_json(orient="records")

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
    SELECT c.instructor_name, c.subject, c.course_number, c.title, s.term_name, s.term_id
    FROM courses c JOIN semesters s ON c.term_id = s.term_id
    WHERE c.instructor_name != '' AND c.instructor_name != 'TBA'
    ORDER BY s.term_id
""", conn2)

inst_data = {}
for name, group in df_inst_detail.groupby('instructor_name'):
    top_courses = group.groupby(['subject', 'course_number']).agg(
        title=('title', 'first'), times=('term_id', 'count')).reset_index().nlargest(10, 'times')
    inst_data[name] = {
        't': int(len(group)),
        's': sorted(group['subject'].unique().tolist()),
        'n': int(group['term_id'].nunique()),
        'f': group['term_name'].iloc[0],
        'l': group['term_name'].iloc[-1],
        'c': [[r['subject'], r['course_number'], r['title'], int(r['times'])] for _, r in top_courses.iterrows()]
    }
inst_explorer_json = json.dumps(inst_data)

conn2.close()

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VisualizeAUS — 20 Years of AUS Course Data</title>
<meta name="description" content="Interactive visualizations of 20 years of course data from the American University of Sharjah.">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;500;600;700;800&family=Montserrat:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg: #faf7f2;
  --bg-warm: #f3ede3;
  --bg-card: #ffffff;
  --bg-hero: #010508;
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
  --radius: 12px;
  --radius-sm: 8px;
  --font-display: 'Raleway', sans-serif;
  --font-body: 'Montserrat', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --max-width: 1100px;
}}

html {{ scroll-behavior: smooth; }}

body {{
  font-family: var(--font-body);
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  font-size: 16px;
  font-weight: 400;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}}

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
  background: rgba(250, 247, 242, 0.92);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
}}

nav .nav-inner {{
  max-width: var(--max-width);
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 0;
  overflow-x: auto;
  scrollbar-width: none;
  padding: 0 2rem;
}}

nav .nav-inner::-webkit-scrollbar {{ display: none; }}

nav a {{
  color: var(--text-muted);
  text-decoration: none;
  font-size: 0.82rem;
  font-weight: 500;
  white-space: nowrap;
  transition: color 0.2s;
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
}}

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
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  margin-bottom: 1.5rem;
}}

.chart-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}}

/* ===================== EXPLANATIONS ===================== */
.explanation {{
  background: var(--accent-bg);
  border: 1px solid #f0dbd8;
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
@media (max-width: 768px) {{
  .chart-row {{ grid-template-columns: 1fr; }}
  .stats-row {{ flex-wrap: wrap; }}
  .stat {{ padding: 1rem 0.75rem; }}
  .stat-value {{ font-size: 1.3rem; }}
  .hero h1 {{ font-size: 2.25rem; }}
  section {{ padding: 3rem 0; }}
  .section-header h2 {{ font-size: 1.6rem; }}
  nav a {{ padding: 0.75rem 0.6rem; font-size: 0.75rem; }}
  .container {{ padding: 0 1rem; }}
}}

</style>
</head>
<body>

<!-- Hero -->
<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">American University of Sharjah</div>
    <h1>VisualizeAUS</h1>
    <p class="subtitle">
      Twenty years of course data, visualized and made explorable.
      Every section, instructor, prerequisite, and schedule from 2005 to 2026.
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
  <div class="nav-inner">
    <a href="#" class="nav-brand">VisualizeAUS</a>
    <a href="#growth">Growth</a>
    <a href="#subjects">Subjects</a>
    <a href="#levels">Levels</a>
    <a href="#instructors">Instructors</a>
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
</nav>

<div class="container">

<!-- 1. Growth -->
<section id="growth">
  <div class="section-header">
    <div class="section-num">01 &mdash; University Growth</div>
    <h2>Two Decades of Expansion</h2>
    <p>How has AUS grown its course offerings from 2005 to 2026?</p>
  </div>
  <div class="chart-container">{charts['growth']}</div>
  <div class="explanation">
    Each dot represents one semester. <strong>Red dots are Fall semesters</strong>, blue are Spring, and green are Summer terms. The dashed line shows the overall upward trend. AUS has grown from about 1,100 course sections per regular semester in 2005 to nearly 2,000 in 2025 &mdash; a <strong>{growth_pct:.0f}% increase</strong>. The peak was <strong>{peak_sem}</strong> with {peak_val:,} sections. Summer terms are much smaller (200-400 sections) and appear as the lower cluster.
  </div>
  <div class="chart-container">{charts['courses_vs_sections']}</div>
  <div class="explanation">
    The blue line tracks total sections offered, while the red line tracks unique courses. Both have grown, but total sections grew faster &mdash; meaning AUS is offering <strong>more sections of existing courses</strong> (to accommodate more students) in addition to introducing new ones.
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
    Mathematics (MTH) has the most sections of any subject &mdash; over 5,500 across 20 years &mdash; because nearly every student takes multiple math courses. The next largest subjects are Civil Engineering (CVE), Mechanical Engineering (MCE), and Electrical Engineering (ELE), reflecting AUS's strong engineering focus.
  </div>
  <div class="chart-container">{charts['subject_lines']}</div>
  <div class="explanation">
    This shows how the <strong>top 10 subjects</strong> have evolved semester by semester. Most subjects show steady or growing offerings. Engineering subjects tend to grow in step with each other, suggesting coordinated program expansion.
  </div>
  <div class="chart-container">{charts['subject_heatmap']}</div>
  <div class="explanation">
    Each cell shows the number of sections a subject offered in a given year. <strong>Darker red means more sections.</strong> You can see the overall growth pattern clearly &mdash; most subjects get darker as you move from left to right.
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
    This stacked area chart shows how enrollment across different academic levels has evolved over 20 years. <strong>Undergraduate sections dominate at {undergrad_pct:.0f}%</strong> of all offerings. The Graduate program has been present since 2005, while the <strong>Doctorate program launched in 2019</strong> and the Achievement Academy started in 2011. Notice how all levels have grown over time, with the undergraduate base expanding steadily.
  </div>
  <div class="chart-container">{charts['levels_dist']}</div>
  <div class="explanation">
    The distribution across all semesters shows the massive dominance of undergraduate education at AUS, with graduate sections being the second largest category at {grad_total:,} total sections. Specialized programs like the Doctorate and Achievement Academy are smaller but growing.
  </div>
  <div class="chart-container">{charts['levels_by_subject']}</div>
  <div class="explanation">
    This reveals which subjects serve multiple academic levels. Subjects at the top have the highest proportion of graduate-level sections. Some subjects like MBA and certain engineering programs serve a significant graduate population, while others like MTH and PHY are almost exclusively undergraduate.
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
    The most prolific instructor has taught <strong>nearly 500 sections</strong> over their career at AUS. The color indicates how many semesters they've been active &mdash; darker blue means a longer tenure.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['tenure']}</div>
    <div class="chart-container">{charts['active_instructors']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> Most instructors teach for a relatively short time &mdash; the histogram is heavily skewed toward 1-5 semesters. However, a significant number have been active for 20+ semesters (10+ years). <strong>Right:</strong> The number of active instructors has grown from about 300 in 2005 to over 500 in recent years.
  </div>
  <div class="chart-container">{charts['instructor_diversity']}</div>
  <div class="explanation">
    Each bubble represents an instructor. The x-axis shows total sections taught, the y-axis shows how many distinct subjects they teach, and the size reflects their tenure. <strong>Instructors in the upper-right are both prolific and versatile</strong> &mdash; teaching many sections across multiple subjects. Most instructors cluster in the lower-left (few sections, 1-2 subjects), while a handful of "superstar" instructors stand out.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['tba_rate']}</div>
    <div class="chart-container">{charts['instructor_workload']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> The TBA rate shows the percentage of sections each semester where no instructor was assigned at scrape time. High TBA in recent semesters often means instructors haven't been finalized yet. <strong>Right:</strong> The average number of sections per instructor per semester has remained relatively stable, hovering around 3-4 sections, showing consistent workload distribution.
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
    This chart shows the evolution of teaching methods at AUS. <strong>Traditional (in-person) instruction dominates</strong> across almost every semester. The most dramatic shift occurred during <strong>COVID-19 (2020-2021)</strong>, when non-traditional delivery surged. After the pandemic, AUS largely returned to traditional methods, though some non-traditional instruction persists.
  </div>
  <div class="chart-container">{charts['modality_by_subject']}</div>
  <div class="explanation">
    Not all subjects adopted non-traditional teaching equally. This chart shows which subjects had the highest proportion of non-traditional sections across all time. Some subjects naturally lend themselves to online/blended formats, while others (especially lab-heavy engineering and science courses) remained largely in-person.
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
    Total sections barely dipped: Fall 2019 had {f19_sections} sections, Fall 2020 had {f20_sections} ({covid_section_change:+.1f}%). The real story is what came <em>after</em> &mdash; AUS surged to {latest_sections} sections by {latest_term} ({growth_since:+.1f}% vs Fall 2019). Course variety (red dashed line) dipped more noticeably, falling {abs(course_variety_drop):.1f}% by Spring 2021 as the university consolidated offerings while maintaining section counts.
  </div>
  <div class="chart-container">{charts['covid_variety']}</div>
  <div class="explanation">
    Instructor count fell from {f19_inst} (Fall 2019) to {f20_inst} (Fall 2020), a {abs(inst_change):.1f}% decline. Combined with fewer unique courses, this suggests AUS kept sections running by concentrating teaching among fewer instructors on a narrower set of courses &mdash; a resilience strategy, not a collapse.
  </div>
  <div class="chart-container">{charts['covid_subjects']}</div>
  <div class="explanation">
    Each cell shows a subject's section count indexed to <strong>Fall 2019 = 100</strong>. Green cells above 100 mean growth; red below 100 means contraction. During Fall 2020, {subjects_shrank} subjects shrank by 10%+, {subjects_grew} grew by 10%+, and {subjects_stable} held steady. Language programs (ELP, ARA) and media (MCM) were hit hardest, while computing (CMP, COE) and sustainability (ESM) expanded &mdash; reflecting a shift toward technical subjects during the pandemic.
  </div>
  <div class="chart-container">{charts['covid_classrooms']}</div>
  <div class="explanation">
    The most lasting COVID effect: sections without an assigned physical classroom rose from {precovid_noroom}% (Fall 2019) to {latest_noroom}% ({latest_term}). This didn't spike during COVID itself, but climbed steadily afterward &mdash; suggesting a permanent structural shift toward flexible or unassigned scheduling that outlasted the pandemic.
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
    This heatmap shows <strong>how many course sections are scheduled at each day-time combination</strong> across all 20 years. The busiest slots are <strong>Monday and Wednesday around 11:00 AM and 2:00 PM</strong>. Sunday through Thursday are the main teaching days (the UAE work week).
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['day_patterns']}</div>
    <div class="chart-container">{charts['buildings']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> The two dominant scheduling patterns are <strong>Mon/Wed (MW)</strong> and <strong>Tue/Thu/Sun (TRU)</strong>, together accounting for over half of all sections. <strong>Right:</strong> New Academic Building 1 hosts the most sections, followed by the Language Building and Engineering Building Right.
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
    This shows how many completely new courses were introduced each year. The peak was <strong>{peak_new_year}</strong> with <strong>{peak_new_count} new courses</strong>. The early years (2005-2008) show high counts because the database starts in 2005 &mdash; many courses that already existed appear as "new" in the first captured year. After that baseline, the rate of new course introduction reveals genuine curriculum expansion.
  </div>
  <div class="chart-container">{charts['course_longevity']}</div>
  <div class="explanation">
    How long do courses survive in the catalog? <strong>{one_sem_courses} courses ({one_sem_pct}%) were offered in only one semester</strong> &mdash; these are likely special topics, experimental courses, or one-off offerings. Meanwhile, <strong>{veteran_courses} courses have been offered for 30+ semesters</strong> (15+ years), forming the stable core of the AUS curriculum.
  </div>
  <div class="chart-container">{charts['most_consistent']}</div>
  <div class="explanation">
    These are the marathon runners of the AUS curriculum &mdash; courses that have been offered the most semesters. Foundational courses in math, English, physics, and engineering dominate this list, as they serve the widest student populations.
  </div>
  <div class="chart-container">{charts['courses_discontinued']}</div>
  <div class="explanation">
    This shows courses that were last offered before 2020 (having previously been offered for at least 5 semesters). These are <strong>{total_discontinued} courses that appear to have been discontinued</strong> &mdash; removed from the active curriculum. Spikes in certain years may correspond to department restructuring or program changes.
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
    Red bars show how many other courses list this course as a prerequisite, while blue bars show how many prerequisites it requires. Foundational courses like intro math, physics, and programming have enormous outgoing connections.
  </div>

  <h3 class="chains-title">Longest Prerequisite Chains</h3>
  <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">These are the longest sequences where each course requires the previous one. A chain of {len(longest_chains[0][1]) if longest_chains else 'N/A'} courses means a student must pass {len(longest_chains[0][1]) - 1 if longest_chains else 'N/A'} prerequisite courses before reaching the final one.</p>
  <div class="chains-grid">
  {''.join('<div class="chain"><div class="chain-header">' + str(len(path)) + ' courses &middot; ' + path[-1] + '</div>' + ''.join('<div class="chain-connector"></div><div class="chain-step"><span class="chain-step-num">' + str(i+1) + '</span>' + c + '</div>' for i, c in enumerate(path)) + '</div>' for _, path in longest_chains[:6])}
  </div>

  <div class="chart-container" style="margin-top: 2rem">{charts['coe_network']}</div>
  <div class="explanation">
    This interactive network graph shows all <strong>Computer Engineering (COE) courses</strong> and their prerequisites. Red nodes are COE courses; blue nodes are prerequisites from other departments. Hover over nodes to see course names.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['prereq_complexity']}</div>
    <div class="chart-container">{charts['cross_dept']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> Departments ranked by average prerequisites per course. Engineering and science departments have the most complex prerequisite structures. <strong>Right:</strong> A matrix showing which departments depend on which others for prerequisites. Many engineering departments depend heavily on MTH and PHY courses.
  </div>

  <h3 class="chains-title">Corequisite Analysis</h3>
  <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">Corequisites are courses that must be taken simultaneously. AUS has <strong>{total_coreq_links:,}</strong> corequisite links across the curriculum.</p>
  {'<div class="chart-container">' + charts.get("coreq_top", "") + '</div>' if "coreq_top" in charts else ""}
  <div class="explanation">
    The most common corequisite pairs are typically lecture-lab combinations (e.g., a physics lecture with its corresponding lab). These mandatory pairings ensure students get both theoretical and practical instruction simultaneously.
  </div>
  {'<div class="chart-container">' + charts.get("coreq_vs_prereq", "") + '</div>' if "coreq_vs_prereq" in charts else ""}
  <div class="explanation">
    This compares the number of prerequisite vs corequisite requirements by department. Most departments rely more heavily on prerequisites, but some (especially lab-intensive programs) have significant corequisite requirements.
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
    Each bar shows how many prerequisite links require that minimum grade. <strong>C- dominates overwhelmingly</strong> as the university-wide standard passing grade. The second most common is <strong>C (no minus)</strong>, followed by <strong>A-</strong> &mdash; used in competitive programs.
  </div>
  <div class="chart-container">{charts['grade_strictness']}</div>
  <div class="explanation">
    Each department's bar shows the <strong>percentage breakdown</strong> of grade levels required for its prerequisites. Departments on the left have the highest proportion of strict requirements (A or B range). Green (C-) dominates for most departments.
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
    Course attributes are tags that classify courses for general education requirements, major electives, and special designations. <strong>"Preparatory"</strong> and <strong>"MTH Major Elective"</strong> are the most common tags. Science requirements, communication courses, and social science requirements round out the top categories &mdash; reflecting AUS's broad general education structure.
  </div>
  <div class="chart-container">{charts['attributes_over_time']}</div>
  <div class="explanation">
    This stacked area chart shows how different categories of course attributes have evolved over time. Growth in "Communication/English" and "Natural Sciences" tags reflects expanding gen-ed requirements. The overall increase in tagged sections mirrors the university's growth in total offerings.
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
    <strong>Left:</strong> The vast majority of AUS courses are 3-credit courses. Labs, independent studies, and capstones often differ from the 3-credit standard. <strong>Right:</strong> Blue bars show lecture hours and red bars show lab hours by department. Engineering and science departments have significantly more lab hours.
  </div>
  <div class="chart-container">{charts['lecture_lab']}</div>
  <div class="explanation">
    The stacked bars show absolute counts of lab vs lecture sections, while the green line shows the percentage of labs. The lab percentage has remained fairly stable at around 15-20%.
  </div>
</section>

<!-- 13. Enrollment -->
<section id="enrollment">
  <div class="section-header">
    <div class="section-num">13 &mdash; Enrollment</div>
    <h2>How Full Are Classes?</h2>
    <p>Tracking seat availability and enrollment restrictions across 20 years.</p>
  </div>
  <div class="chart-container">{charts['enrollment']}</div>
  <div class="explanation">
    Each bar represents a semester, split into sections with <strong>available seats (green)</strong> and sections <strong>completely full (red)</strong>. Note that this reflects a snapshot when scraped, not the full registration period.
  </div>
  <div class="chart-container">{charts['fill_rate']}</div>
  <div class="explanation">
    Subjects ranked by what percentage of their sections were full. <strong>Higher bars mean more sections at capacity</strong> and potentially high demand.
  </div>
  {'<div class="chart-container">' + charts.get("fees", "") + '</div><div class="explanation">Course fees vary by college and type. The boxes show the spread of fee amounts. Different colleges charge different technology fee tiers.</div>' if "fees" in charts else ""}
  {'<div class="chart-container">' + charts.get("fee_trend", "") + '</div><div class="explanation">Fee trends over time reflect general cost inflation and evolving technology requirements across different colleges.</div>' if "fee_trend" in charts else ""}
  <div class="chart-container">{charts['restriction_types']}</div>
  <div class="explanation">
    <strong>{restricted_pct}% of sections have enrollment restrictions.</strong> The vast majority are level-based restrictions (e.g., "Undergraduate Only" or "Graduate Only"), ensuring students are in appropriate courses for their academic level. A smaller portion restrict by major, college, or class standing.
  </div>
</section>

<!-- 14. Browse Data -->
<section id="browse">
  <div class="section-header">
    <div class="section-num">14 &mdash; Browse &amp; Explore</div>
    <h2>Explore the Dataset</h2>
    <p>Search courses, look up instructors, and explore prerequisite chains interactively.</p>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('courses')">Recent Courses</div>
    <div class="tab" onclick="switchTab('catalog')">Course Catalog</div>
    <div class="tab" onclick="switchTab('dep-explorer')">Dependency Explorer</div>
    <div class="tab" onclick="switchTab('inst-explorer')">Instructor Lookup</div>
  </div>

  <div id="tab-courses" class="tab-content active">
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="course-search" placeholder="Search courses &mdash; try COE, Calculus, or an instructor name..." oninput="filterTable('courses')">
        <select id="course-semester" onchange="filterTable('courses')">
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

  <div id="tab-catalog" class="tab-content">
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="catalog-search" placeholder="Search catalog &mdash; try a subject, keyword, or department..." oninput="filterTable('catalog')">
      </div>
      <div class="table-scroll">
        <table id="catalog-table">
          <thead>
            <tr>
              <th onclick="sortTable('catalog', 0)">Subject</th>
              <th onclick="sortTable('catalog', 1)">Number</th>
              <th onclick="sortTable('catalog', 2)">Description</th>
              <th onclick="sortTable('catalog', 3)">Credits</th>
              <th onclick="sortTable('catalog', 4)">Lecture</th>
              <th onclick="sortTable('catalog', 5)">Lab</th>
              <th onclick="sortTable('catalog', 6)">Department</th>
            </tr>
          </thead>
          <tbody id="catalog-body"></tbody>
        </table>
      </div>
      <div class="table-info" id="catalog-info"></div>
    </div>
  </div>

  <div id="tab-dep-explorer" class="tab-content">
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="dep-search" placeholder="Type a course code (e.g., COE 221, MTH 104, PHY 101)..." oninput="searchDeps(this.value)" autocomplete="off">
      </div>
      <div id="dep-suggestions" style="display:none; background: var(--bg-card); border: 1px solid var(--border); border-top: 0; border-radius: 0 0 var(--radius-sm) var(--radius-sm); max-height: 200px; overflow-y: auto;"></div>
      <div id="dep-result" style="padding: 1.5rem;">
        <p style="color: var(--text-muted); font-size: 0.9rem;">Search for any course to see its complete dependency tree: prerequisites (with minimum grades), corequisites, and which courses need it as a prerequisite.</p>
      </div>
    </div>
  </div>

  <div id="tab-inst-explorer" class="tab-content">
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="inst-search" placeholder="Type an instructor name..." oninput="searchInst(this.value)" autocomplete="off">
      </div>
      <div id="inst-suggestions" style="display:none; background: var(--bg-card); border: 1px solid var(--border); border-top: 0; border-radius: 0 0 var(--radius-sm) var(--radius-sm); max-height: 200px; overflow-y: auto;"></div>
      <div id="inst-result" style="padding: 1.5rem;">
        <p style="color: var(--text-muted); font-size: 0.9rem;">Search for any instructor to see their complete teaching history: courses taught, subjects, tenure, and career timeline at AUS.</p>
      </div>
    </div>
  </div>
</section>

</div>

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
<script>
// ---- Data ----
const courseData = {browse_json};
const catalogData = {catalog_json};
const depData = {dep_explorer_json};
const instData = {inst_explorer_json};

// Populate semester dropdown
const semesters = [...new Set(courseData.map(r => r.term_name))];
const semSelect = document.getElementById('course-semester');
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
    const sem = document.getElementById('course-semester').value;
    if (sem) data = data.filter(r => r.term_name === sem);
  }}

  if (search) {{
    data = data.filter(r => Object.values(r).some(v =>
      v && String(v).toLowerCase().includes(search)));
  }}

  renderTable(type, data);
}}

let sortState = {{}};
function sortTable(type, col) {{
  const key = type + col;
  sortState[key] = !(sortState[key] || false);
  const asc = sortState[key];

  let data = type === 'courses' ? [...courseData] : [...catalogData];
  const keys = type === 'courses'
    ? ['subject','course_number','title','instructor_name','days','start_time','classroom','term_name']
    : ['subject','course_number','description','credit_hours','lecture_hours','lab_hours','department'];

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
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}}

// ---- Dependency Explorer ----
function searchDeps(query) {{
  const q = query.toUpperCase().trim();
  const sugBox = document.getElementById('dep-suggestions');
  if (q.length < 2) {{ sugBox.style.display = 'none'; return; }}
  const matches = Object.keys(depData).filter(k => k.includes(q)).sort().slice(0, 12);
  if (matches.length === 0) {{ sugBox.style.display = 'none'; return; }}
  sugBox.style.display = 'block';
  sugBox.innerHTML = matches.map(m =>
    `<div style="padding: 0.5rem 1rem; cursor: pointer; font-family: var(--font-mono); font-size: 0.85rem; border-bottom: 1px solid var(--border-light);"
          onmouseover="this.style.background='var(--bg-warm)'" onmouseout="this.style.background=''"
          onclick="showDeps('${{m}}')">${{m}}</div>`).join('');
}}

function showDeps(course) {{
  document.getElementById('dep-suggestions').style.display = 'none';
  document.getElementById('dep-search').value = course;
  const data = depData[course];
  const result = document.getElementById('dep-result');
  if (!data) {{ result.innerHTML = '<p style="color: var(--text-muted);">Course not found in dependency graph.</p>'; return; }}

  let html = `<h3 style="font-family: var(--font-display); margin-bottom: 1rem; color: var(--accent);">${{course}}</h3>`;

  // Prerequisites
  html += '<div style="margin-bottom: 1.5rem;"><h4 style="font-family: var(--font-mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.5rem;">Prerequisites (' + data.p.length + ')</h4>';
  if (data.p.length === 0) {{ html += '<p style="color: var(--text-light); font-size: 0.9rem;">None</p>'; }}
  else {{ html += '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">' + data.p.map(p =>
    `<span style="background: var(--accent-bg); border: 1px solid #f0dbd8; padding: 0.3rem 0.7rem; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.82rem; cursor: pointer;" onclick="showDeps('${{p.c}}')">${{p.c}}${{p.g ? ' (min: ' + p.g + ')' : ''}}</span>`
  ).join('') + '</div>'; }}
  html += '</div>';

  // Corequisites
  const coreqs = data.q || [];
  html += '<div style="margin-bottom: 1.5rem;"><h4 style="font-family: var(--font-mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.5rem;">Corequisites (' + coreqs.length + ')</h4>';
  if (coreqs.length === 0) {{ html += '<p style="color: var(--text-light); font-size: 0.9rem;">None</p>'; }}
  else {{ html += '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">' + coreqs.map(c =>
    `<span style="background: #f3e8f9; border: 1px solid #d5b8e8; padding: 0.3rem 0.7rem; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.82rem; cursor: pointer;" onclick="showDeps('${{c}}')">${{c}}</span>`
  ).join('') + '</div>'; }}
  html += '</div>';

  // Needed by
  html += '<div style="margin-bottom: 1.5rem;"><h4 style="font-family: var(--font-mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.5rem;">Is Prerequisite For (' + data.n.length + ')</h4>';
  if (data.n.length === 0) {{ html += '<p style="color: var(--text-light); font-size: 0.9rem;">No courses depend on this</p>'; }}
  else {{ html += '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">' + data.n.map(c =>
    `<span style="background: #e8f5e9; border: 1px solid #a5d6a7; padding: 0.3rem 0.7rem; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.82rem; cursor: pointer;" onclick="showDeps('${{c}}')">${{c}}</span>`
  ).join('') + '</div>'; }}
  html += '</div>';

  // Full prerequisite chain (recursive)
  function getChain(c, visited) {{
    if (!depData[c] || visited.has(c)) return [c];
    visited.add(c);
    if (depData[c].p.length === 0) return [c];
    const longest = depData[c].p.reduce((best, p) => {{
      const chain = getChain(p.c, visited);
      return chain.length > best.length ? chain : best;
    }}, []);
    return [...longest, c];
  }}
  const chain = getChain(course, new Set());
  if (chain.length > 1) {{
    html += '<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);"><h4 style="font-family: var(--font-mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.75rem;">Longest Prerequisite Chain (' + chain.length + ' courses)</h4>';
    html += '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem;">';
    chain.forEach((c, i) => {{
      if (i > 0) html += '<span style="color: var(--accent); font-weight: 700;">&rarr;</span>';
      const isCurrent = c === course;
      html += `<span style="background: ${{isCurrent ? 'var(--accent)' : 'var(--bg-warm)'}}; color: ${{isCurrent ? '#fff' : 'var(--text)'}}; padding: 0.25rem 0.6rem; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.8rem; cursor: pointer;" onclick="showDeps('${{c}}')">${{c}}</span>`;
    }});
    html += '</div></div>';
  }}

  result.innerHTML = html;
}}

// ---- Instructor Explorer ----
function searchInst(query) {{
  const q = query.toLowerCase().trim();
  const sugBox = document.getElementById('inst-suggestions');
  if (q.length < 2) {{ sugBox.style.display = 'none'; return; }}
  const matches = Object.keys(instData).filter(k => k.toLowerCase().includes(q)).sort().slice(0, 12);
  if (matches.length === 0) {{ sugBox.style.display = 'none'; return; }}
  sugBox.style.display = 'block';
  sugBox.innerHTML = matches.map(m =>
    `<div style="padding: 0.5rem 1rem; cursor: pointer; font-size: 0.85rem; border-bottom: 1px solid var(--border-light);"
          onmouseover="this.style.background='var(--bg-warm)'" onmouseout="this.style.background=''"
          onclick="showInst('${{m.replace(/'/g, "\\\\'")}}')"><strong>${{m}}</strong></div>`).join('');
}}

function showInst(name) {{
  document.getElementById('inst-suggestions').style.display = 'none';
  document.getElementById('inst-search').value = name;
  const data = instData[name];
  const result = document.getElementById('inst-result');
  if (!data) {{ result.innerHTML = '<p style="color: var(--text-muted);">Instructor not found.</p>'; return; }}

  let html = `<h3 style="font-family: var(--font-display); margin-bottom: 0.5rem; color: var(--accent);">${{name}}</h3>`;

  // Stats row
  html += '<div style="display: flex; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap;">';
  html += `<div><span style="font-family: var(--font-mono); font-size: 1.5rem; font-weight: 600; color: var(--text);">${{data.t}}</span><br><span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em;">Sections</span></div>`;
  html += `<div><span style="font-family: var(--font-mono); font-size: 1.5rem; font-weight: 600; color: var(--text);">${{data.n}}</span><br><span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em;">Semesters</span></div>`;
  html += `<div><span style="font-family: var(--font-mono); font-size: 1.5rem; font-weight: 600; color: var(--text);">${{data.s.length}}</span><br><span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em;">Subjects</span></div>`;
  html += '</div>';

  // Tenure
  html += `<p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;"><strong>Active:</strong> ${{data.f}} &mdash; ${{data.l}}</p>`;

  // Subjects
  html += '<div style="margin-bottom: 1.5rem;"><h4 style="font-family: var(--font-mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.5rem;">Subjects Taught</h4>';
  html += '<div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">' + data.s.map(s =>
    `<span style="background: var(--bg-warm); border: 1px solid var(--border); padding: 0.2rem 0.6rem; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.82rem;">${{s}}</span>`
  ).join('') + '</div></div>';

  // Top courses
  html += '<div><h4 style="font-family: var(--font-mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.5rem;">Top Courses (by times taught)</h4>';
  html += '<table style="width: 100%; font-size: 0.85rem;"><thead><tr><th style="text-align: left; padding: 0.4rem; font-weight: 600; color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; border-bottom: 1px solid var(--border);">Course</th><th style="text-align: left; padding: 0.4rem; font-weight: 600; color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; border-bottom: 1px solid var(--border);">Title</th><th style="text-align: right; padding: 0.4rem; font-weight: 600; color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; border-bottom: 1px solid var(--border);">Times</th></tr></thead><tbody>';
  data.c.forEach(c => {{
    html += `<tr><td style="padding: 0.4rem; font-family: var(--font-mono); font-size: 0.82rem;">${{c[0]}} ${{c[1]}}</td><td style="padding: 0.4rem; color: var(--text-secondary);">${{c[2]}}</td><td style="padding: 0.4rem; text-align: right; font-family: var(--font-mono); font-weight: 600;">${{c[3]}}</td></tr>`;
  }});
  html += '</tbody></table></div>';

  result.innerHTML = html;
}}

// Initial render
renderTable('courses', courseData);
renderTable('catalog', catalogData);

// Active nav on scroll
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('nav a[href^="#"]');
window.addEventListener('scroll', () => {{
  let current = '';
  sections.forEach(s => {{
    if (window.scrollY >= s.offsetTop - 200) current = s.id;
  }});
  navLinks.forEach(a => {{
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  }});
}}, {{ passive: true }});
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
