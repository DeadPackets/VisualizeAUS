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
    <img src="https://img.shields.io/badge/charts-45-C4972F?style=flat-square" alt="45 charts">
    <br/>
    <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Plotly-interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
    <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
  </p>
</p>

---

An interactive data visualization dashboard exploring every course offered at the **American University of Sharjah** from 2005 to 2026. Built with Plotly for 45 fully interactive charts, searchable data tables, interactive course dependency explorer, instructor career lookup, and a prerequisite network of 152,968 links.

**Data source:** [AUSCrawl](https://github.com/DeadPackets/AUSCrawl)

---

## What's Inside

45 interactive visualizations across 14 sections:

| Section | What You'll Find |
|---------|-----------------|
| **University Growth** | Course sections per semester with trend analysis, unique courses vs total sections |
| **Subject Analysis** | Top 25 subjects, subject evolution over time, heatmap across 20 years |
| **Academic Levels** | Undergraduate/Graduate/Doctorate distribution over time, level mix by subject |
| **Instructor Analysis** | Top 30 instructors, tenure distribution, teaching diversity scatter, workload trends |
| **Teaching Modality** | Traditional vs non-traditional instruction, modality by subject |
| **COVID-19 Impact** | Pandemic disruption analysis: section counts, modality shift, lab impact, subject-level changes |
| **Schedule Patterns** | Day-time heatmap, day pattern popularity, building utilization |
| **Curriculum Evolution** | New courses per year, course longevity, most consistent courses, discontinued courses |
| **Prerequisite Network** | Most connected courses, longest chains, COE network graph, corequisite analysis |
| **Grade Requirements** | Minimum grade distribution, department strictness comparison |
| **Course Attributes** | Gen-Ed tag distribution, attribute categories over time |
| **Course Catalog** | Credit hours distribution, lecture vs lab hours by department |
| **Enrollment** | Seat availability, fill rates, fee analysis, enrollment restriction types |
| **Browse & Explore** | Searchable tables, **interactive dependency explorer**, **instructor career lookup** |

---

## View the Site

Visit the live site at: **http://projects.deadpackets.pw/VisualizeAUS/**

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
  analysis.ipynb       # Jupyter notebook with full analysis
  build.py             # Static site generator
  aus_courses.db       # Downloaded at build time (not in repo)
  _site/
    index.html         # Generated site (4.5 MB, self-contained)
  .github/
    workflows/
      deploy.yml       # GitHub Pages deployment
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
