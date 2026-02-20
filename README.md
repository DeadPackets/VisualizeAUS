<p align="center">
  <h1 align="center">VisualizeAUS</h1>
  <p align="center">
    <strong>20 years of AUS course data, beautifully visualized.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/courses-73%2C418-C4972F?style=flat-square" alt="73,418 courses">
    <img src="https://img.shields.io/badge/semesters-98-C4972F?style=flat-square" alt="98 semesters">
    <img src="https://img.shields.io/badge/instructors-1%2C649-C4972F?style=flat-square" alt="1,649 instructors">
    <img src="https://img.shields.io/badge/prerequisites-152%2C968-C4972F?style=flat-square" alt="152,968 prerequisites">
    <img src="https://img.shields.io/badge/charts-49-C4972F?style=flat-square" alt="49 charts">
    <br/>
    <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Plotly-interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
    <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
    <img src="https://img.shields.io/badge/mobile-responsive-C4972F?style=flat-square" alt="Mobile Responsive">
  </p>
</p>

---

An interactive data visualization dashboard exploring every course offered at the **American University of Sharjah** from 2005 to 2026. 49 fully interactive Plotly charts across 14 analytical sections, searchable data tables, an interactive course dependency explorer, instructor career lookup, and a prerequisite network of 152,968 links — all in a single self-contained HTML page.

**Data source:** [AUSCrawl](https://github.com/DeadPackets/AUSCrawl)

---

## What's Inside

49 interactive visualizations across 14 sections, plus 2 interactive explorers:

| Section | Charts | What You'll Find |
|---------|:------:|-----------------|
| **University Growth** | 2 | Semester-by-semester section count with polynomial trend line, unique courses vs total sections divergence |
| **Subject Analysis** | 3 | Top 25 subjects by section count, top 10 subject trajectories over time, full subject-by-year heatmap across 20 years |
| **Academic Levels** | 3 | Undergraduate/Graduate/Doctorate stacked area over time, overall level distribution, graduate percentage by subject |
| **Instructor Analysis** | 10 | Top 30 most prolific instructors, tenure length distribution, active instructor count over time, teaching diversity scatter (sections vs subjects vs tenure), TBA instructor rate, average workload trends, semester-by-semester recruitment vs departures, retention survival curves by hiring cohort, course ownership scatter (continuity vs turnover), longest instructor-course pairings |
| **Teaching Modality** | 2 | Traditional vs non-traditional instruction evolution, non-traditional rate by subject |
| **COVID-19 Impact** | 4 | Section count resilience through the pandemic (with annotated COVID band), instructor and course variety contraction, per-subject indexed heatmap (Fall 2019 = 100), unassigned classroom trend revealing lasting structural shift |
| **Schedule Patterns** | 5 | Day-time combination heatmap, day pattern popularity distribution, building utilization rankings, Saturday class disappearance (collapsed from 200+ to near-zero by 2010), day pattern evolution stacked area over 20 years |
| **Curriculum Evolution** | 4 | New courses introduced per year, course longevity distribution, most consistently offered courses across 20 years, discontinued courses timeline |
| **Prerequisite Network** | 6 | Most connected courses (prerequisites + dependents), interactive COE prerequisite network graph, prerequisite complexity by department, cross-department dependency matrix, top corequisite pairs, corequisite vs prerequisite comparison by department |
| **Grade Requirements** | 2 | Minimum grade distribution (C- and C dominate), department strictness comparison (stacked percentage) |
| **Course Attributes** | 2 | Gen-Ed and major attribute tag distribution, attribute category evolution over time |
| **Course Catalog** | 3 | Credit hours distribution, lecture vs lab section ratio with trend, lab and lecture hours breakdown by department |
| **Enrollment** | 3 | Seat availability vs full sections per semester, fill rate by subject, enrollment restriction type breakdown (level, major, college, class standing) |
| **Browse & Explore** | — | Searchable course table (73K+ rows, filterable by semester), searchable catalog table, **interactive dependency explorer** (prerequisite chains, corequisites, reverse dependencies), **instructor career lookup** (tenure, subjects, top courses) |

---

## View the Site

Visit the live site at: **http://projects.deadpackets.pw/VisualizeAUS/**

The site is fully responsive and works on mobile devices.

Or build locally:

```bash
git clone https://github.com/DeadPackets/VisualizeAUS
cd VisualizeAUS

# Download the database from AUSCrawl
curl -L -o aus_courses.db https://github.com/DeadPackets/AUSCrawl/raw/master/aus_courses.db

# Install dependencies and build
pip install pandas numpy plotly networkx
python build.py

# Open _site/index.html in your browser
```

---

## Project Structure

```
VisualizeAUS/
  build.py             # Static site generator (~2,400 lines)
  aus_courses.db       # Downloaded at build time (not in repo)
  _site/
    index.html         # Generated site (~4.7 MB, self-contained)
  .github/
    workflows/
      deploy.yml       # GitHub Pages auto-deployment
```

---

## The Database

The `aus_courses.db` file contains 10 normalized tables with 73,418 course sections, 1,649 instructors, 3,007 catalog entries, 71,754 section details, and 152,968 prerequisite/corequisite links. See [AUSCrawl](https://github.com/DeadPackets/AUSCrawl) for the full schema documentation.

---

<p align="center">
  <sub>Built for AUS students, by an AUS student.</sub>
  <br/>
  <a href="https://github.com/DeadPackets/AUSCrawl">AUSCrawl</a> · <a href="https://github.com/DeadPackets/VisualizeAUS/blob/main/LICENSE">MIT License</a>
</p>
