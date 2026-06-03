<p align="center">
  <h1 align="center">VisualizeAUS</h1>
  <p align="center">
    <strong>20 years of AUS course data, beautifully visualized.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/sections-75%2C467-C4972F?style=flat-square" alt="75,467 sections">
    <img src="https://img.shields.io/badge/semesters-101-C4972F?style=flat-square" alt="101 semesters">
    <img src="https://img.shields.io/badge/instructors-1%2C987-C4972F?style=flat-square" alt="1,987 instructors">
    <img src="https://img.shields.io/badge/dependencies-156%2C512-C4972F?style=flat-square" alt="156,512 dependencies">
    <img src="https://img.shields.io/badge/charts-62-C4972F?style=flat-square" alt="62 charts">
    <br/>
    <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Plotly-interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
    <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
    <img src="https://img.shields.io/badge/mobile-responsive-C4972F?style=flat-square" alt="Mobile Responsive">
  </p>
</p>

---

An interactive data visualization dashboard exploring every course offered at the **American University of Sharjah** from 2005 to 2026. 62 fully interactive Plotly charts across 14 analytical sections, a cross-linked **Course Explorer** (full profiles + visual prerequisite roadmaps), instructor lookup, a **Degree Explorer**, searchable data tables, and a prerequisite network of 156,512 links — a single static page that runs entirely in your browser, no backend.

**Data source:** [AUSCrawl](https://github.com/DeadPackets/AUSCrawl)

---

## What's Inside

62 interactive visualizations across 14 sections, plus 2 interactive explorers:

| Section | Charts | What You'll Find |
|---------|:------:|-----------------|
| **University Growth** | 2 | Semester-by-semester section count with polynomial trend line, unique courses vs total sections divergence |
| **Subject Analysis** | 3 | Top 25 subjects by section count, top 10 subject trajectories over time, full subject-by-year heatmap across 20 years |
| **Academic Levels** | 3 | Undergraduate/Graduate/Doctorate stacked area over time, overall level distribution, graduate percentage by subject |
| **Instructor Analysis** | 13 | Top 30 most prolific instructors, tenure length distribution, active instructor count over time, teaching diversity scatter (sections vs subjects vs tenure), TBA instructor rate, average workload trends, semester-by-semester recruitment vs departures, retention survival curves by hiring cohort, course ownership scatter (continuity vs turnover), longest instructor-course pairings, **plus Team Teaching**: co-instruction rate over time, co-teaching rate by department, and the most frequent co-teaching partnerships |
| **Teaching Modality** | 2 | Traditional vs non-traditional instruction evolution, non-traditional rate by subject |
| **COVID-19 Impact** | 4 | Section count resilience through the pandemic (with annotated COVID band), instructor and course variety contraction, per-subject indexed heatmap (Fall 2019 = 100), unassigned classroom trend revealing lasting structural shift |
| **Schedule Patterns** | 5 | Day-time combination heatmap, day pattern popularity distribution, building utilization rankings, Saturday class disappearance (collapsed from 200+ to near-zero by 2010), day pattern evolution stacked area over 20 years |
| **Curriculum Evolution** | 4 | New courses introduced per year, course longevity distribution, most consistently offered courses across 20 years, discontinued courses timeline |
| **Prerequisite Network** | 11 | Most connected courses (prerequisites + dependents), interactive COE prerequisite network graph, prerequisite complexity by department, cross-department dependency matrix, top corequisite pairs, corequisite vs prerequisite comparison by department, **plus Prerequisite Logic** (from the parsed AND/OR boolean trees): requirement shape distribution, logic nesting depth, toughest gateways by mandatory prerequisites, curriculum flexibility by department, and concurrent-enrollment prevalence over time |
| **Grade Requirements** | 2 | Minimum grade distribution (C- and C dominate), department strictness comparison (stacked percentage) |
| **Course Attributes** | 4 | Gen-Ed and major attribute tag distribution, attribute category evolution over time, **plus Degree Requirement Mapping**: programs with the most elective course options, and the most reusable courses across degree programs |
| **Course Catalog** | 3 | Credit hours distribution, lecture vs lab section ratio with trend, lab and lecture hours breakdown by department |
| **Enrollment & Access** | 4 | Section closure per semester (open vs closed seats; completed terms only), closure rate by subject, typed include/exclude enrollment restrictions, and the courses most often gated to a specific major/college/program. **Note:** Banner exposes only a binary open/closed flag, never seat counts — these track section *closure*, not a true fill rate |
| **Browse & Explore** | — | **Course Explorer** — search any course for its full profile (credits, animated sections-per-year chart, usual times, instructors), a visual **AND/OR prerequisite roadmap**, corequisites, what it unlocks, and which degree programs it counts toward; **Instructor Lookup** — career-activity timeline, teaching-by-subject breakdown, tenure, and top courses; **Degree Explorer** — every course tagged toward a major/minor with a subject-composition chart; and searchable course + catalog tables with live filter-responsive summary bars. Everything is cross-linked — one click jumps between courses, instructors, and programs |

---

## View the Site

Visit the live site at: **http://projects.deadpackets.pw/VisualizeAUS/**

The site is fully responsive and works on mobile devices.

Or build locally:

```bash
git clone https://github.com/DeadPackets/VisualizeAUS
cd VisualizeAUS

# Download the database from the AUSCrawl GitHub Release (gzipped, ~16 MB)
curl -fL -o aus_courses.db.gz \
  https://github.com/DeadPackets/AUSCrawl/releases/latest/download/aus_courses.db.gz
gunzip aus_courses.db.gz

# Install dependencies, run the tests, and build
pip install pandas numpy plotly networkx pytest
pytest -q
python build.py

# Open _site/index.html in your browser
```

---

## Project Structure

```
VisualizeAUS/
  build.py             # Static site generator (~2,900 lines)
  analysis.py          # Pure helpers for the prerequisite/restriction/attribute JSON
  tests/
    test_analysis.py   # Unit tests for analysis.py (run with pytest)
  aus_courses.db       # Downloaded at build time (not in repo)
  _site/
    index.html         # Generated page shell (~0.15 MB — small enough for social crawlers)
    data.js            # Chart + explorer data payload (~7 MB, loaded by index.html)
  .github/
    workflows/
      deploy.yml       # GitHub Pages auto-deployment
```

---

## The Database

The `aus_courses.db` file contains 11 normalized tables with 75,467 course sections, 1,987 instructors, 3,046 catalog entries, ~73,800 section details, and 156,512 prerequisite/corequisite links — plus `section_instructors` (every instructor per section, including co-teachers) and `catalog_detail` (course-level attributes, schedule types, and degree-requirement tags). The dashboard also reads the parsed boolean prerequisite trees (`prerequisites_json`) for its prerequisite-logic analysis. See [AUSCrawl](https://github.com/DeadPackets/AUSCrawl) for the full schema documentation.

---

<p align="center">
  <sub>Built for AUS students, by an AUS student.</sub>
  <br/>
  <a href="https://github.com/DeadPackets/AUSCrawl">AUSCrawl</a> · <a href="https://github.com/DeadPackets/VisualizeAUS/blob/main/LICENSE">MIT License</a>
</p>
