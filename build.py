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
                       div_id=chart_id, config={"responsive": True})


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
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg: #06060b;
  --bg-elevated: #0c0c14;
  --bg-card: #0f0f19;
  --bg-card-hover: #14141f;
  --border: rgba(255,255,255,0.06);
  --border-strong: rgba(255,255,255,0.1);
  --border-glow: rgba(196, 151, 47, 0.2);
  --text: #e8e8ec;
  --text-secondary: #a1a1aa;
  --text-muted: #63636e;
  --text-dim: #3f3f46;
  --gold: #C4972F;
  --gold-light: #dbb456;
  --gold-pale: #f0d88a;
  --gold-glow: rgba(196, 151, 47, 0.08);
  --gold-glow-strong: rgba(196, 151, 47, 0.15);
  --radius: 16px;
  --radius-sm: 10px;
  --font-display: 'DM Serif Display', Georgia, serif;
  --font-body: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}}

html {{ scroll-behavior: smooth; }}

body {{
  font-family: var(--font-body);
  background: var(--bg);
  color: var(--text);
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  overflow-x: hidden;
}}

/* ---- Grain overlay ---- */
body::after {{
  content: '';
  position: fixed;
  inset: 0;
  z-index: 9999;
  pointer-events: none;
  opacity: 0.025;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  background-repeat: repeat;
  background-size: 256px 256px;
}}

/* ---- Scroll reveal ---- */
.reveal {{
  opacity: 0;
  transform: translateY(32px);
  transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1),
              transform 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}}
.reveal.visible {{
  opacity: 1;
  transform: translateY(0);
}}
.reveal-delay-1 {{ transition-delay: 0.1s; }}
.reveal-delay-2 {{ transition-delay: 0.2s; }}
.reveal-delay-3 {{ transition-delay: 0.3s; }}

/* ===================== HERO ===================== */
.hero {{
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 2rem;
  position: relative;
  overflow: hidden;
}}

.hero::before {{
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 50% at 50% 40%, var(--gold-glow-strong) 0%, transparent 70%),
    radial-gradient(ellipse 40% 30% at 30% 60%, rgba(59,130,246,0.04) 0%, transparent 70%),
    radial-gradient(ellipse 40% 30% at 70% 70%, rgba(168,85,247,0.03) 0%, transparent 70%);
  animation: heroGlow 12s ease-in-out infinite alternate;
}}

@keyframes heroGlow {{
  0% {{ opacity: 0.6; }}
  100% {{ opacity: 1; }}
}}

/* Geometric grid pattern */
.hero::after {{
  content: '';
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(var(--border) 1px, transparent 1px),
    linear-gradient(90deg, var(--border) 1px, transparent 1px);
  background-size: 80px 80px;
  mask-image: radial-gradient(ellipse 60% 60% at 50% 50%, black 20%, transparent 70%);
  -webkit-mask-image: radial-gradient(ellipse 60% 60% at 50% 50%, black 20%, transparent 70%);
  opacity: 0.5;
}}

.hero-content {{
  position: relative;
  z-index: 1;
  max-width: 860px;
}}

.hero-eyebrow {{
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--gold);
  letter-spacing: 0.25em;
  text-transform: uppercase;
  margin-bottom: 1.5rem;
  opacity: 0;
  animation: fadeUp 0.8s 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}

.hero h1 {{
  font-family: var(--font-display);
  font-size: clamp(3.25rem, 8vw, 6.5rem);
  font-weight: 400;
  font-style: italic;
  letter-spacing: -0.02em;
  line-height: 1.05;
  color: var(--text);
  margin-bottom: 1.75rem;
  opacity: 0;
  animation: fadeUp 0.8s 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}

.hero h1 em {{
  font-style: normal;
  background: linear-gradient(135deg, var(--gold) 0%, var(--gold-pale) 60%, var(--gold-light) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}

.hero .subtitle {{
  font-size: 1.15rem;
  color: var(--text-secondary);
  font-weight: 300;
  max-width: 560px;
  margin: 0 auto 3.5rem;
  line-height: 1.7;
  opacity: 0;
  animation: fadeUp 0.8s 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}

@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(24px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

.stats-row {{
  display: flex;
  justify-content: center;
  gap: 0;
  opacity: 0;
  animation: fadeUp 0.8s 0.65s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}

.stat {{
  padding: 1.75rem 2.5rem;
  text-align: center;
  position: relative;
}}

.stat:not(:last-child)::after {{
  content: '';
  position: absolute;
  right: 0;
  top: 25%;
  height: 50%;
  width: 1px;
  background: var(--border-strong);
}}

.stat-value {{
  font-family: var(--font-mono);
  font-size: 2rem;
  font-weight: 600;
  color: var(--gold);
  letter-spacing: -0.03em;
  line-height: 1;
}}

.stat-label {{
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin-top: 0.5rem;
  font-weight: 500;
}}

.scroll-cue {{
  position: absolute;
  bottom: 2.5rem;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-dim);
  font-size: 0.7rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-family: var(--font-mono);
  animation: fadeUp 0.8s 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  opacity: 0;
}}

.scroll-cue::after {{
  content: '';
  width: 1px;
  height: 32px;
  background: linear-gradient(to bottom, var(--text-dim), transparent);
  animation: scrollPulse 2s ease-in-out infinite;
}}

@keyframes scrollPulse {{
  0%, 100% {{ opacity: 0.3; transform: scaleY(1); }}
  50% {{ opacity: 0.8; transform: scaleY(1.3); }}
}}

/* ===================== NAV ===================== */
nav {{
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(6, 6, 11, 0.8);
  backdrop-filter: blur(24px) saturate(1.2);
  -webkit-backdrop-filter: blur(24px) saturate(1.2);
  border-bottom: 1px solid var(--border);
}}

nav .nav-inner {{
  max-width: 1280px;
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
  font-size: 0.8rem;
  font-weight: 500;
  white-space: nowrap;
  transition: color 0.25s, border-color 0.25s;
  padding: 1rem 1.25rem;
  border-bottom: 2px solid transparent;
  letter-spacing: 0.01em;
}}

nav a:hover {{ color: var(--text-secondary); }}

nav a.active {{
  color: var(--gold);
  border-bottom-color: var(--gold);
}}

nav .nav-brand {{
  font-family: var(--font-display);
  font-style: italic;
  font-weight: 400;
  color: var(--text);
  font-size: 1.05rem;
  padding-right: 1.5rem;
  margin-right: 0.5rem;
  border-right: 1px solid var(--border);
  letter-spacing: -0.01em;
}}

/* ===================== SECTIONS ===================== */
.container {{
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 2rem;
}}

section {{
  padding: 6rem 0;
  position: relative;
}}

section + section {{
  border-top: 1px solid var(--border);
}}

.section-header {{
  margin-bottom: 3.5rem;
  max-width: 720px;
}}

.section-header .section-num {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 500;
  color: var(--gold);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}}

.section-header .section-num::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, var(--gold-glow-strong), transparent);
  max-width: 120px;
}}

.section-header h2 {{
  font-family: var(--font-display);
  font-size: 2.75rem;
  font-weight: 400;
  letter-spacing: -0.02em;
  color: var(--text);
  margin-bottom: 1rem;
  line-height: 1.15;
}}

.section-header p {{
  color: var(--text-secondary);
  font-size: 1.05rem;
  line-height: 1.75;
  font-weight: 300;
}}

/* ===================== CHARTS ===================== */
.chart-container {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  margin-bottom: 2rem;
  position: relative;
  transition: border-color 0.4s, box-shadow 0.4s;
}}

.chart-container:hover {{
  border-color: var(--border-glow);
  box-shadow: 0 0 40px -10px var(--gold-glow);
}}

.chart-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}}

/* ===================== INSIGHT ===================== */
.insight {{
  background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 1.5rem 2rem;
  margin: 2rem 0;
  font-size: 0.95rem;
  line-height: 1.75;
  color: var(--text-secondary);
}}

.insight strong {{
  color: var(--gold-light);
  font-weight: 600;
}}

/* ===================== TABLES ===================== */
.table-wrapper {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin: 2rem 0;
}}

.table-controls {{
  display: flex;
  gap: 0.75rem;
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  background: var(--bg-elevated);
}}

.table-controls input, .table-controls select {{
  background: var(--bg);
  border: 1px solid var(--border-strong);
  color: var(--text);
  padding: 0.6rem 1rem;
  border-radius: var(--radius-sm);
  font-family: var(--font-body);
  font-size: 0.85rem;
  outline: none;
  transition: border-color 0.25s, box-shadow 0.25s;
}}

.table-controls input:focus, .table-controls select:focus {{
  border-color: var(--gold);
  box-shadow: 0 0 0 3px var(--gold-glow);
}}

.table-controls input::placeholder {{ color: var(--text-dim); }}
.table-controls input {{ flex: 1; min-width: 200px; }}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.83rem;
}}

thead th {{
  background: var(--bg-elevated);
  padding: 0.85rem 1rem;
  text-align: left;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  position: sticky;
  top: 0;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  transition: color 0.2s;
  font-family: var(--font-mono);
}}

thead th:hover {{ color: var(--gold); }}

tbody tr {{
  border-bottom: 1px solid var(--border);
  transition: background 0.2s;
}}

tbody tr:hover {{ background: var(--bg-card-hover); }}

tbody td {{
  padding: 0.65rem 1rem;
  color: var(--text-secondary);
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.table-scroll {{
  max-height: 520px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--border-strong) transparent;
}}

.table-scroll::-webkit-scrollbar {{ width: 6px; }}
.table-scroll::-webkit-scrollbar-track {{ background: transparent; }}
.table-scroll::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 3px; }}

.table-info {{
  padding: 0.85rem 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.78rem;
  color: var(--text-dim);
  font-family: var(--font-mono);
}}

/* ===================== PREREQ CHAINS ===================== */
.chains-title {{
  font-family: var(--font-display);
  color: var(--text);
  margin: 3rem 0 1.25rem;
  font-size: 1.5rem;
  font-weight: 400;
}}

.chain {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1rem 1.5rem;
  margin: 0.6rem 0;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  overflow-x: auto;
  white-space: nowrap;
  transition: border-color 0.3s;
}}

.chain:hover {{ border-color: var(--border-glow); }}
.chain-steps {{ color: var(--gold); margin-right: 1rem; font-weight: 600; }}
.chain-arrow {{ color: var(--text-dim); margin: 0 0.3rem; }}

/* ===================== TABS ===================== */
.tabs {{
  display: flex;
  gap: 0;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--border);
}}

.tab {{
  padding: 0.85rem 1.75rem;
  cursor: pointer;
  color: var(--text-muted);
  font-size: 0.85rem;
  font-weight: 500;
  border-bottom: 2px solid transparent;
  transition: color 0.25s, border-color 0.25s;
  font-family: var(--font-body);
}}

.tab:hover {{ color: var(--text-secondary); }}
.tab.active {{ color: var(--gold); border-bottom-color: var(--gold); }}

.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* ===================== FOOTER ===================== */
footer {{
  text-align: center;
  padding: 5rem 2rem;
  color: var(--text-dim);
  font-size: 0.85rem;
  border-top: 1px solid var(--border);
  position: relative;
}}

footer .footer-brand {{
  font-family: var(--font-display);
  font-style: italic;
  font-size: 1.5rem;
  color: var(--text-muted);
  margin-bottom: 1rem;
}}

footer a {{ color: var(--gold); text-decoration: none; transition: color 0.2s; }}
footer a:hover {{ color: var(--gold-light); }}

footer p {{ margin-top: 0.4rem; }}

/* ===================== RESPONSIVE ===================== */
@media (max-width: 768px) {{
  .chart-row {{ grid-template-columns: 1fr; }}
  .stats-row {{ flex-wrap: wrap; }}
  .stat {{ padding: 1.25rem 1.5rem; }}
  .hero h1 {{ font-size: 2.75rem; }}
  section {{ padding: 4rem 0; }}
  .section-header h2 {{ font-size: 2rem; }}
  nav .nav-inner {{ padding: 0 1rem; }}
  nav a {{ padding: 0.85rem 0.75rem; font-size: 0.75rem; }}
}}

@media (max-width: 480px) {{
  .stat {{ padding: 1rem; }}
  .stat-value {{ font-size: 1.5rem; }}
  .container {{ padding: 0 1rem; }}
}}

</style>
</head>
<body>

<!-- Hero -->
<div class="hero">
  <div class="hero-content">
    <div class="hero-eyebrow">American University of Sharjah</div>
    <h1><em>VisualizeAUS</em></h1>
    <p class="subtitle">
      Twenty years of course data, visualized and made explorable.
      Every section, instructor, prerequisite, and schedule
      from 2005 to 2026.
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
  <div class="scroll-cue">Scroll</div>
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
  <div class="section-header reveal">
    <div class="section-num">01 — University Growth</div>
    <h2>Two Decades of Expansion</h2>
    <p>AUS has grown steadily since 2005, expanding from around 1,100 course sections per regular semester to nearly 2,000.</p>
  </div>
  <div class="chart-container reveal reveal-delay-1">{charts['growth']}</div>
  <div class="insight reveal reveal-delay-2">
    <strong>{growth_pct:.0f}% growth</strong> in regular semester sections from Spring 2005 to Spring 2026.
    Peak semester: <strong>{peak_sem}</strong> with {peak_val:,} sections.
    Average trend: <strong>+{z[0]:.1f} sections</strong> per semester.
  </div>
  <div class="chart-container reveal">{charts['courses_vs_sections']}</div>
</section>

<!-- 2. Subjects -->
<section id="subjects">
  <div class="section-header reveal">
    <div class="section-num">02 — Subject Analysis</div>
    <h2>What Does AUS Teach?</h2>
    <p>{table_counts['subjects']} subject areas spanning engineering, sciences, arts, and humanities. The top 10 subjects account for nearly half of all sections.</p>
  </div>
  <div class="chart-container reveal">{charts['subjects_bar']}</div>
  <div class="chart-container reveal">{charts['subject_lines']}</div>
  <div class="chart-container reveal">{charts['subject_heatmap']}</div>
</section>

<!-- 3. Instructors -->
<section id="instructors">
  <div class="section-header reveal">
    <div class="section-num">03 — Instructor Analysis</div>
    <h2>The Teaching Workforce</h2>
    <p>{table_counts['instructors']:,} unique instructors have taught at AUS since 2005. Some have been active for over a decade.</p>
  </div>
  <div class="chart-container reveal">{charts['instructors']}</div>
  <div class="chart-row">
    <div class="chart-container reveal">{charts['tenure']}</div>
    <div class="chart-container reveal reveal-delay-1">{charts['active_instructors']}</div>
  </div>
  <div class="chart-container reveal">{charts['tba_rate']}</div>
</section>

<!-- 4. Schedule -->
<section id="schedule">
  <div class="section-header reveal">
    <div class="section-num">04 — Schedule Patterns</div>
    <h2>When Does AUS Have Class?</h2>
    <p>AUS follows a UAE schedule with classes Sunday through Thursday. The dominant patterns are Mon/Wed 75-minute blocks and Tue/Thu/Sun 50-minute blocks.</p>
  </div>
  <div class="chart-container reveal">{charts['schedule_heatmap']}</div>
  <div class="chart-row">
    <div class="chart-container reveal">{charts['day_patterns']}</div>
    <div class="chart-container reveal reveal-delay-1">{charts['buildings']}</div>
  </div>
</section>

<!-- 5. Prerequisites -->
<section id="prerequisites">
  <div class="section-header reveal">
    <div class="section-num">05 — Prerequisite Network</div>
    <h2>The Dependency Web</h2>
    <p>{G.number_of_nodes()} courses connected by {G.number_of_edges()} prerequisite edges. Some chains span {len(longest_chains[0][1]) if longest_chains else 'N/A'} courses deep.</p>
  </div>
  <div class="chart-container reveal">{charts['prereq_connected']}</div>

  <h3 class="chains-title reveal">Longest Prerequisite Chains</h3>
  {''.join(f'<div class="chain reveal"><span class="chain-steps">[{len(path)}]</span> {"<span class=chain-arrow> &rarr; </span>".join(path)}</div>' for _, path in longest_chains)}

  <div class="chart-container reveal" style="margin-top: 2.5rem">{charts['coe_network']}</div>
  <div class="chart-container reveal">{charts['prereq_complexity']}</div>
  <div class="chart-container reveal">{charts['cross_dept']}</div>
</section>

<!-- 6. Grades -->
<section id="grades">
  <div class="section-header reveal">
    <div class="section-num">06 — Grade Requirements</div>
    <h2>Academic Rigor</h2>
    <p>Most prerequisites require a minimum grade of C-, but some departments are significantly stricter.</p>
  </div>
  <div class="chart-container reveal">{charts['grades']}</div>
  <div class="chart-container reveal">{charts['grade_strictness']}</div>
</section>

<!-- 7. Catalog -->
<section id="catalog">
  <div class="section-header reveal">
    <div class="section-num">07 — Course Catalog</div>
    <h2>Credits, Lectures, and Labs</h2>
    <p>{table_counts['catalog']:,} unique courses in the catalog. The vast majority are 3-credit courses with lecture-only format.</p>
  </div>
  <div class="chart-row">
    <div class="chart-container reveal">{charts['credit_hours']}</div>
    <div class="chart-container reveal reveal-delay-1">{charts['lab_lecture']}</div>
  </div>
  <div class="chart-container reveal">{charts['lecture_lab']}</div>
</section>

<!-- 8. Enrollment -->
<section id="enrollment">
  <div class="section-header reveal">
    <div class="section-num">08 — Enrollment</div>
    <h2>How Full Are Classes?</h2>
    <p>Tracking seat availability across 20 years reveals which subjects and semesters face the most capacity pressure.</p>
  </div>
  <div class="chart-container reveal">{charts['enrollment']}</div>
  <div class="chart-container reveal">{charts['fill_rate']}</div>
  {'<div class="chart-container reveal">' + charts.get("fees", "") + '</div>' if "fees" in charts else ""}
  {'<div class="chart-container reveal">' + charts.get("fee_trend", "") + '</div>' if "fee_trend" in charts else ""}
</section>

<!-- 9. Browse Data -->
<section id="browse">
  <div class="section-header reveal">
    <div class="section-num">09 — Browse Data</div>
    <h2>Explore the Dataset</h2>
    <p>Search, filter, and sort through the raw course data. Showing the most recent 5,000 sections and the full course catalog.</p>
  </div>

  <div class="tabs reveal">
    <div class="tab active" onclick="switchTab('courses')">Recent Courses</div>
    <div class="tab" onclick="switchTab('catalog')">Course Catalog</div>
  </div>

  <div id="tab-courses" class="tab-content active">
    <div class="table-wrapper reveal">
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
  <div class="footer-brand">VisualizeAUS</div>
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
// ---- Scroll reveal observer ----
const revealObserver = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (entry.isIntersecting) {{
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }}
  }});
}}, {{ threshold: 0.08, rootMargin: '0px 0px -40px 0px' }});

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

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
