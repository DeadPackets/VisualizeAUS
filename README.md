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
    <br/>
    <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Plotly-interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
    <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
  </p>
</p>

---

An interactive data visualization dashboard exploring every course offered at the **American University of Sharjah** from 2005 to 2026. Built with Plotly for fully interactive charts, searchable data tables, and a dependency graph of 152,968 prerequisite links.

**Data source:** [AUSCrawl](https://github.com/DeadPackets/AUSCrawl)

---

## What's Inside

25 interactive visualizations across 9 sections:

| Section | What You'll Find |
|---------|-----------------|
| **University Growth** | Course sections per semester with trend analysis, unique courses vs total sections |
| **Subject Analysis** | Top 25 subjects, subject evolution over time, heatmap across 20 years |
| **Instructor Analysis** | Top 30 instructors, tenure distribution, active faculty over time, TBA rate |
| **Schedule Patterns** | Day-time heatmap, day pattern popularity, building utilization |
| **Prerequisite Network** | Most connected courses, longest chains, COE network graph, cross-department dependencies |
| **Grade Requirements** | Minimum grade distribution, department strictness comparison |
| **Course Catalog** | Credit hours distribution, lecture vs lab hours by department |
| **Enrollment** | Seat availability trends, subject fill rates, fee analysis |
| **Browse Data** | Searchable, sortable tables for 5,000 recent courses and 3,007 catalog entries |

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
    index.html         # Generated site (3.3 MB, self-contained)
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
