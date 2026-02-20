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
color_map = {"Fall": "#e74c3c", "Spring": "#3498db", "Summer": "#2ecc71",
             "Summer II": "#27ae60", "Wintermester": "#9b59b6"}
for sem_type in ["Fall", "Spring", "Summer", "Summer II", "Wintermester"]:
    mask = df_growth["semester_type"] == sem_type
    if not mask.any():
        continue
    fig.add_trace(go.Scatter(
        x=df_growth.loc[mask, "term_name"], y=df_growth.loc[mask, "total_sections"],
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
                  xaxis=dict(tickangle=-45, dtick=4),
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
top15_subjects = df_subjects.head(15)["subject"].tolist()
df_top15 = df_subj_time[df_subj_time["subject"].isin(top15_subjects)].copy()
fig = px.line(df_top15, x="term_name", y="sections", color="subject",
              title="Top 15 Subjects: Section Count Over Time",
              labels={"sections": "Sections", "term_name": "Semester"},
              color_discrete_sequence=px.colors.qualitative.Set3)
fig.update_layout(template=TEMPLATE, height=550, xaxis=dict(tickangle=-45, dtick=4),
                  legend=dict(title="Subject"))
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
             color="semesters_active", color_continuous_scale="Blues",
             hover_data=["subjects_taught", "semesters_active"],
             title="Top 30 Instructors by Total Sections Taught",
             labels={"total_sections": "Total Sections", "instructor_name": "",
                     "semesters_active": "Semesters Active"})
fig.update_layout(template=TEMPLATE, height=700)
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
                color_continuous_scale="YlOrRd",
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
fig = px.bar(df_grades, x="minimum_grade", y="count",
             title="Minimum Grade Requirements Across All Prerequisites",
             labels={"minimum_grade": "Minimum Grade", "count": "Dependencies"},
             color="count", color_continuous_scale="RdYlGn_r", text="count")
fig.update_traces(textposition="outside")
fig.update_layout(template=TEMPLATE, height=400)
fig.update_coloraxes(showscale=False)
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
fig = go.Figure()
for col, color, label in [("grade_A", "#e74c3c", "A/A-"), ("grade_B", "#f39c12", "B range"),
    ("grade_C", "#3498db", "C/C+"), ("grade_C_minus", "#2ecc71", "C-"), ("grade_D", "#95a5a6", "D range")]:
    fig.add_trace(go.Bar(x=df_strict["subject"], y=df_strict[col], name=label, marker_color=color))
fig.update_layout(template=TEMPLATE, title="Grade Requirement Strictness by Department",
                  barmode="stack", height=500, xaxis_title="Subject", yaxis_title="Dependencies",
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
df_dept_hours = df_dept_hours[df_dept_hours["course_count"] >= 10].sort_values("avg_lab", ascending=False)
df_dept_hours["dept_short"] = df_dept_hours["department"].str.replace(" Department", "").str.replace(" (n.a.)", "", regex=False)
fig = go.Figure()
fig.add_trace(go.Bar(y=df_dept_hours["dept_short"], x=df_dept_hours["avg_lecture"],
    name="Lecture Hours", orientation="h", marker_color="#3498db"))
fig.add_trace(go.Bar(y=df_dept_hours["dept_short"], x=df_dept_hours["avg_lab"],
    name="Lab Hours", orientation="h", marker_color="#e74c3c"))
fig.update_layout(template=TEMPLATE, title="Average Lecture vs Lab Hours by Department",
                  barmode="stack", height=600, xaxis_title="Hours",
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
    fig = px.box(df_fees, x="fee_type", y="amount",
                 title="Fee Amount Distribution by Type",
                 labels={"amount": "Amount (AED)", "fee_type": "Fee Type"}, color="fee_type")
    fig.update_layout(template=TEMPLATE, height=450, showlegend=False)
    charts["fees"] = chart_html(fig, "fees")

    df_fee_trend = df_fees.groupby(["term_id", "term_name", "fee_type"]).agg(
        avg_amount=("amount", "mean"), count=("amount", "count")).reset_index().sort_values("term_id")
    top_fees = df_fees["fee_type"].value_counts().head(5).index.tolist()
    df_fee_trend_top = df_fee_trend[df_fee_trend["fee_type"].isin(top_fees)]
    fig = px.line(df_fee_trend_top, x="term_name", y="avg_amount", color="fee_type",
                  title="Average Fee Amount Over Time",
                  labels={"avg_amount": "Amount (AED)", "term_name": "Semester"}, markers=True)
    fig.update_layout(template=TEMPLATE, height=450, xaxis=dict(tickangle=-45, dtick=4))
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

# Longest prereq chains
longest_chains = []
try:
    longest = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            longest[node] = [node]
        else:
            best = max((longest.get(p, [p]) for p in preds), key=len)
            longest[node] = best + [node]
    sorted_paths = sorted(longest.items(), key=lambda x: len(x[1]), reverse=True)
    longest_chains = sorted_paths[:12]
except nx.NetworkXUnfeasible:
    pass

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

.chain {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.85rem 1.25rem;
  margin: 0.5rem 0;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  overflow-x: auto;
  white-space: nowrap;
}}

.chain-steps {{ color: var(--accent); margin-right: 0.75rem; font-weight: 600; }}
.chain-arrow {{ color: var(--text-light); margin: 0 0.2rem; }}

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
    </div>
  </div>
</div>

<!-- Navigation -->
<nav>
  <div class="nav-inner">
    <a href="#" class="nav-brand">VisualizeAUS</a>
    <a href="#growth">Growth</a>
    <a href="#subjects">Subjects</a>
    <a href="#instructors">Instructors</a>
    <a href="#schedule">Schedule</a>
    <a href="#prerequisites">Prerequisites</a>
    <a href="#grades">Grades</a>
    <a href="#catalog">Catalog</a>
    <a href="#enrollment">Enrollment</a>
    <a href="#browse">Browse</a>
  </div>
</nav>

<div class="container">

<!-- 1. Growth -->
<section id="growth">
  <div class="section-header">
    <div class="section-num">01 — University Growth</div>
    <h2>Two Decades of Expansion</h2>
    <p>How has AUS grown its course offerings from 2005 to 2026?</p>
  </div>
  <div class="chart-container">{charts['growth']}</div>
  <div class="explanation">
    Each dot represents one semester. <strong>Red dots are Fall semesters</strong>, blue are Spring, and green are Summer terms. The dashed line shows the overall upward trend. AUS has grown from about 1,100 course sections per regular semester in 2005 to nearly 2,000 in 2025 — a <strong>{growth_pct:.0f}% increase</strong>. The peak was <strong>{peak_sem}</strong> with {peak_val:,} sections. Summer terms are much smaller (200-400 sections) and appear as the lower cluster.
  </div>
  <div class="chart-container">{charts['courses_vs_sections']}</div>
  <div class="explanation">
    The blue line tracks total sections offered, while the red line tracks unique courses. Both have grown, but total sections grew faster — meaning AUS is offering <strong>more sections of existing courses</strong> (to accommodate more students) in addition to introducing new ones. On average, each unique course has about 2.2 sections per semester.
  </div>
</section>

<!-- 2. Subjects -->
<section id="subjects">
  <div class="section-header">
    <div class="section-num">02 — Subject Analysis</div>
    <h2>What Does AUS Teach?</h2>
    <p>{table_counts['subjects']} subject areas spanning engineering, sciences, arts, and humanities.</p>
  </div>
  <div class="chart-container">{charts['subjects_bar']}</div>
  <div class="explanation">
    Mathematics (MTH) has the most sections of any subject — over 5,500 across 20 years — because nearly every student at AUS takes multiple math courses regardless of major. The next largest subjects are Civil Engineering (CVE), Mechanical Engineering (MCE), and Electrical Engineering (ELE), reflecting AUS's strong engineering focus. The <strong>top 10 subjects account for nearly half</strong> of all sections ever offered.
  </div>
  <div class="chart-container">{charts['subject_lines']}</div>
  <div class="explanation">
    This shows how the top 15 subjects have evolved semester by semester. Most subjects show steady or growing offerings. Some interesting patterns: <strong>Writing (WRI)</strong> courses saw a significant increase around 2010, likely reflecting curriculum changes. Engineering subjects tend to grow in step with each other, suggesting coordinated program expansion.
  </div>
  <div class="chart-container">{charts['subject_heatmap']}</div>
  <div class="explanation">
    Each cell shows the number of sections a subject offered in a given year. <strong>Darker red means more sections.</strong> You can see the overall growth pattern clearly — most subjects get darker (more sections) as you move from left to right. Gaps or lighter spots can indicate years where a subject reduced offerings or was restructured.
  </div>
</section>

<!-- 3. Instructors -->
<section id="instructors">
  <div class="section-header">
    <div class="section-num">03 — Instructor Analysis</div>
    <h2>The Teaching Workforce</h2>
    <p>{table_counts['instructors']:,} unique instructors have taught at AUS since 2005.</p>
  </div>
  <div class="chart-container">{charts['instructors']}</div>
  <div class="explanation">
    The most prolific instructor has taught <strong>nearly 500 sections</strong> over their career at AUS. The color indicates how many semesters they've been active — darker blue means a longer tenure. Many of the top instructors have been active for 30+ semesters (15+ years), suggesting a stable core faculty.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['tenure']}</div>
    <div class="chart-container">{charts['active_instructors']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> Most instructors teach for a relatively short time — the histogram is heavily skewed toward 1-5 semesters. However, a significant number have been active for 20+ semesters (10+ years), forming the experienced backbone of AUS's faculty. <strong>Right:</strong> The number of active instructors per semester has grown from about 300 in 2005 to over 500 in recent years, tracking the university's overall expansion.
  </div>
  <div class="chart-container">{charts['tba_rate']}</div>
  <div class="explanation">
    The "TBA rate" is the percentage of sections each semester where no instructor was assigned at the time the data was scraped. <strong>A high TBA rate (especially in recent semesters) often means instructors haven't been finalized yet</strong> rather than that sections are truly unstaffed. Spring 2026, for example, may still have high TBA because the semester was upcoming when this data was collected.
  </div>
</section>

<!-- 4. Schedule -->
<section id="schedule">
  <div class="section-header">
    <div class="section-num">04 — Schedule Patterns</div>
    <h2>When Does AUS Have Class?</h2>
    <p>AUS follows a UAE schedule: classes run Sunday through Thursday, with Saturday occasionally used.</p>
  </div>
  <div class="chart-container">{charts['schedule_heatmap']}</div>
  <div class="explanation">
    This heatmap shows <strong>how many course sections are scheduled at each day-time combination</strong> across all 20 years. The busiest slots are clearly visible as deep red: <strong>Monday and Wednesday around 11:00 AM and 2:00 PM</strong> are the most popular. Sunday through Thursday are the main teaching days (the UAE work week). Saturday is rarely used. Notice that 8:00 AM slots are relatively light — early mornings are less popular for scheduling.
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['day_patterns']}</div>
    <div class="chart-container">{charts['buildings']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> The two dominant scheduling patterns are <strong>Mon/Wed (MW)</strong> — typically 75-minute blocks — and <strong>Tue/Thu/Sun (TRU)</strong> — typically 50-minute blocks. Together these account for over half of all sections. <strong>Right:</strong> New Academic Building 1 hosts the most sections by far, followed by the Language Building and Engineering Building Right (EB2). This reflects the campus layout where general-purpose lecture halls are concentrated in NAB1.
  </div>
</section>

<!-- 5. Prerequisites -->
<section id="prerequisites">
  <div class="section-header">
    <div class="section-num">05 — Prerequisite Network</div>
    <h2>The Dependency Web</h2>
    <p>{G.number_of_nodes()} courses connected by {G.number_of_edges()} prerequisite edges.</p>
  </div>
  <div class="chart-container">{charts['prereq_connected']}</div>
  <div class="explanation">
    This chart shows the <strong>most connected courses in the prerequisite graph</strong>. Red bars show how many other courses list this course as a prerequisite ("is prerequisite for"), while blue bars show how many prerequisites the course itself requires. Foundational courses like introductory math, physics, and programming have enormous outgoing connections — dozens of upper-level courses depend on them.
  </div>

  <h3 class="chains-title">Longest Prerequisite Chains</h3>
  <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">These are the longest sequences where each course requires the previous one. A chain of {len(longest_chains[0][1]) if longest_chains else 'N/A'} means a student must pass {len(longest_chains[0][1]) - 1 if longest_chains else 'N/A'} prerequisite courses before reaching the final one.</p>
  {''.join('<div class="chain"><span class="chain-steps">[' + str(len(path)) + ']</span> ' + '<span class=chain-arrow> &rarr; </span>'.join(path) + '</div>' for _, path in longest_chains)}

  <div class="chart-container" style="margin-top: 2rem">{charts['coe_network']}</div>
  <div class="explanation">
    This interactive network graph shows all <strong>Computer Engineering (COE) courses</strong> and their prerequisites. Red nodes are COE courses; blue nodes are prerequisites from other departments (like MTH, PHY, CMP). The size of each node reflects how many connections it has. You can see how foundational courses in math and physics feed into the COE curriculum, and how upper-level COE courses form deep chains. Hover over nodes to see course names.
  </div>
  <div class="chart-container">{charts['prereq_complexity']}</div>
  <div class="explanation">
    This compares departments by how many prerequisites their courses require on average. <strong>A higher bar means more prerequisite requirements per course</strong> in that department. Engineering and science departments tend to have the most complex prerequisite structures, reflecting the sequential nature of technical curricula. Arts and humanities subjects typically have fewer formal prerequisites.
  </div>
  <div class="chart-container">{charts['cross_dept']}</div>
  <div class="explanation">
    This matrix shows <strong>which departments depend on which other departments</strong> for prerequisites. Read it as: courses in the row department require prerequisites from the column department. Strong off-diagonal cells indicate heavy cross-department dependencies. For example, many engineering departments depend heavily on MTH (Mathematics) and PHY (Physics) courses. The darker the blue, the more courses have that cross-department dependency.
  </div>
</section>

<!-- 6. Grades -->
<section id="grades">
  <div class="section-header">
    <div class="section-num">06 — Grade Requirements</div>
    <h2>Academic Rigor</h2>
    <p>What minimum grades do prerequisites require, and how strict are different departments?</p>
  </div>
  <div class="chart-container">{charts['grades']}</div>
  <div class="explanation">
    The overwhelming majority of prerequisites at AUS require a minimum grade of <strong>C-</strong>, which is the standard passing grade for moving forward. However, some courses require higher grades: <strong>C (no minus)</strong> is the second most common, followed by <strong>A-</strong>, which appears in certain competitive programs. A small number of prerequisites require a B or higher — these are typically for advanced courses where strong foundational knowledge is critical.
  </div>
  <div class="chart-container">{charts['grade_strictness']}</div>
  <div class="explanation">
    Each bar is broken down by the grade levels required for that department's prerequisites. Departments shown at the left of the chart have <strong>the highest proportion of strict grade requirements</strong> (A or B range). The green (C-) portion dominates for most departments, confirming that C- is the university-wide standard. Red and orange segments (A/A- and B range) highlight departments with more demanding progression standards.
  </div>
</section>

<!-- 7. Catalog -->
<section id="catalog">
  <div class="section-header">
    <div class="section-num">07 — Course Catalog</div>
    <h2>Credits, Lectures, and Labs</h2>
    <p>{table_counts['catalog']:,} unique courses in the catalog.</p>
  </div>
  <div class="chart-row">
    <div class="chart-container">{charts['credit_hours']}</div>
    <div class="chart-container">{charts['lab_lecture']}</div>
  </div>
  <div class="explanation">
    <strong>Left:</strong> The vast majority of AUS courses are 3-credit courses, which is standard for most universities. A smaller number carry 1, 2, 4, or 6 credits — labs, independent studies, and capstone projects often differ from the 3-credit standard. <strong>Right:</strong> This breaks down lecture versus lab hours by department. Engineering and science departments have significantly more lab hours than humanities departments, reflecting their hands-on curriculum requirements.
  </div>
  <div class="chart-container">{charts['lecture_lab']}</div>
  <div class="explanation">
    This shows the <strong>ratio of lab to lecture sections over time</strong>. The stacked bars show absolute counts, while the green line shows the percentage of sections that are labs. The lab percentage has remained fairly stable at around 15-20%, meaning AUS consistently allocates about one lab section for every five lecture sections.
  </div>
</section>

<!-- 8. Enrollment -->
<section id="enrollment">
  <div class="section-header">
    <div class="section-num">08 — Enrollment</div>
    <h2>How Full Are Classes?</h2>
    <p>Tracking seat availability across 20 years of data.</p>
  </div>
  <div class="chart-container">{charts['enrollment']}</div>
  <div class="explanation">
    Each bar represents a semester, split into sections that had <strong>available seats (green)</strong> and sections that were <strong>completely full (red)</strong>. Note that this data reflects a single snapshot in time (when the database was scraped), not the entire registration period. Early semesters may appear more "full" because enrollment data was captured later in the term, while future semesters may appear more "available" because registration is still ongoing.
  </div>
  <div class="chart-container">{charts['fill_rate']}</div>
  <div class="explanation">
    This ranks subjects by what percentage of their sections were full. <strong>Higher bars mean more sections at capacity.</strong> Subjects with high fill rates are in high demand and may benefit from additional sections. Note that small subjects with few sections can appear disproportionately full due to small sample sizes.
  </div>
  {'<div class="chart-container">' + charts.get("fees", "") + '</div><div class="explanation">Course fees vary by college and type. The boxes show the spread of fee amounts — the line in the middle is the median, and the box covers the 25th to 75th percentile. Different colleges (CAS for Arts & Sciences, CAAD for Architecture, Art and Design) charge different technology fee tiers, with CAAD typically charging more due to specialized software and equipment needs.</div>' if "fees" in charts else ""}
  {'<div class="chart-container">' + charts.get("fee_trend", "") + '</div><div class="explanation">This tracks how the average fee amount has changed over the semesters. Fee increases over time reflect general cost inflation and evolving technology requirements across different colleges.</div>' if "fee_trend" in charts else ""}
</section>

<!-- 9. Browse Data -->
<section id="browse">
  <div class="section-header">
    <div class="section-num">09 — Browse Data</div>
    <h2>Explore the Dataset</h2>
    <p>Search, filter, and sort through the raw data. Showing the most recent 5,000 sections and the full course catalog. Click any column header to sort.</p>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('courses')">Recent Courses</div>
    <div class="tab" onclick="switchTab('catalog')">Course Catalog</div>
  </div>

  <div id="tab-courses" class="tab-content active">
    <div class="table-wrapper">
      <div class="table-controls">
        <input type="text" id="course-search" placeholder="Search courses — try COE, Calculus, or an instructor name..." oninput="filterTable('courses')">
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
        <input type="text" id="catalog-search" placeholder="Search catalog — try a subject, keyword, or department..." oninput="filterTable('catalog')">
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
</section>

</div>

<!-- Footer -->
<footer>
  <p>
    Built with data from <a href="https://github.com/DeadPackets/AUSCrawl">AUSCrawl</a>
    — {table_counts['courses']:,} sections across {table_counts['semesters']} semesters
  </p>
  <p>
    <a href="https://github.com/DeadPackets/VisualizeAUS">GitHub</a>
    &nbsp;&middot;&nbsp; Data scraped from AUS Banner &nbsp;&middot;&nbsp; MIT License
  </p>
</footer>

<script>
// ---- Course data ----
const courseData = {browse_json};
const catalogData = {catalog_json};

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
