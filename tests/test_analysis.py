"""Unit tests for analysis.py — the pure helpers that turn AUSCrawl's
prerequisite/restriction/attribute JSON into chart-ready metrics.

These are kept in a separate module from build.py because importing build.py
runs the entire site generation (it is a procedural script), so its logic
cannot be unit-tested in place. analysis.py is import-safe and pure.
"""

import analysis


# ── prerequisite expression trees ───────────────────────────────────────────
# A node is either a course leaf {"type":"course", ...} or an operator
# {"type":"and"|"or", "operands":[...]}.

SINGLE = {"type": "course", "subject": "WRI", "course_number": "102",
          "min_grade": "C-", "level": "Undergraduate", "concurrent": False}

AND2 = {"type": "and", "operands": [
    {"type": "course", "subject": "MTH", "course_number": "103", "concurrent": False},
    {"type": "course", "subject": "PHY", "course_number": "101", "concurrent": False},
]}

# (A) AND (B OR C) — one mandatory course + a 2-way choice.
AND_WITH_OR = {"type": "and", "operands": [
    {"type": "course", "subject": "MTH", "course_number": "103", "concurrent": False},
    {"type": "or", "operands": [
        {"type": "course", "subject": "PHY", "course_number": "101", "concurrent": False},
        {"type": "course", "subject": "PHY", "course_number": "102", "concurrent": True},
    ]},
]}

OR2 = {"type": "or", "operands": [
    {"type": "course", "subject": "MTH", "course_number": "100", "concurrent": False},
    {"type": "course", "subject": "MTH", "course_number": "101", "concurrent": False},
]}


def test_tree_courses_collects_every_leaf():
    leaves = analysis.tree_courses(AND_WITH_OR)
    codes = sorted(f"{c['subject']} {c['course_number']}" for c in leaves)
    assert codes == ["MTH 103", "PHY 101", "PHY 102"]


def test_tree_has_or():
    assert analysis.tree_has_or(AND_WITH_OR) is True
    assert analysis.tree_has_or(AND2) is False
    assert analysis.tree_has_or(SINGLE) is False


def test_tree_depth_counts_operator_nesting():
    # A bare course is depth 1; a flat and/or is depth 2; nesting adds levels.
    assert analysis.tree_depth(SINGLE) == 1
    assert analysis.tree_depth(AND2) == 2
    assert analysis.tree_depth(AND_WITH_OR) == 3


def test_tree_mandatory_count_ignores_courses_inside_or():
    # Only MTH 103 is unconditionally required; PHY 101/102 are an OR choice.
    assert analysis.tree_mandatory_count(AND_WITH_OR) == 1
    assert analysis.tree_mandatory_count(AND2) == 2
    assert analysis.tree_mandatory_count(SINGLE) == 1
    assert analysis.tree_mandatory_count(OR2) == 0


def test_tree_alternative_paths_multiplies_and_adds():
    # AND multiplies children, OR adds them, a single course is 1 way.
    assert analysis.tree_alternative_paths(SINGLE) == 1
    assert analysis.tree_alternative_paths(AND2) == 1          # 1 * 1
    assert analysis.tree_alternative_paths(OR2) == 2           # 1 + 1
    assert analysis.tree_alternative_paths(AND_WITH_OR) == 2   # 1 * (1 + 1)


def test_tree_has_concurrent():
    assert analysis.tree_has_concurrent(AND_WITH_OR) is True   # PHY 102 concurrent
    assert analysis.tree_has_concurrent(AND2) is False


def test_classify_prereq_shape():
    assert analysis.classify_prereq_shape(SINGLE) == "Single course"
    assert analysis.classify_prereq_shape(AND2) == "All required (AND)"
    assert analysis.classify_prereq_shape(OR2) == "Has alternatives (OR)"
    assert analysis.classify_prereq_shape(AND_WITH_OR) == "Complex (nested AND/OR)"


def test_load_tree_is_safe_on_junk():
    assert analysis.load_tree("") is None
    assert analysis.load_tree("[]") is None
    assert analysis.load_tree("null") is None
    assert analysis.load_tree(None) is None
    assert analysis.load_tree("not json") is None
    assert analysis.load_tree('{"type":"course","subject":"X","course_number":"1"}') is not None


# ── enrollment restrictions (typed include/exclude groups) ───────────────────

def test_restriction_groups_parses_list():
    groups = analysis.restriction_groups(
        '[{"include": true, "type": "Levels", "values": ["Graduate"]}]'
    )
    assert groups == [{"include": True, "type": "Levels", "values": ["Graduate"]}]


def test_restriction_groups_safe_on_junk():
    assert analysis.restriction_groups("") == []
    assert analysis.restriction_groups("null") == []
    assert analysis.restriction_groups(None) == []


def test_restriction_label_uses_include_flag():
    inc = {"include": True, "type": "Majors", "values": []}
    exc = {"include": False, "type": "Colleges", "values": []}
    assert analysis.restriction_label(inc) == "Must be: Major"
    assert analysis.restriction_label(exc) == "Must not be: College"


def test_normalize_restriction_type_shortens_labels():
    assert analysis.normalize_restriction_type("Levels") == "Level"
    assert analysis.normalize_restriction_type("Classifications") == "Class standing"
    assert analysis.normalize_restriction_type(
        "Fields of Study (Major, Minor, or Concentration)") == "Field of study"


# ── restriction *text* parser (full-coverage fallback for restrictions_json) ─
# The typed JSON column is only populated for recently-fetched rows, but the
# raw text is present for ~90% of sections and is highly regular.

def test_restriction_text_groups_parses_include_and_exclude():
    text = ("Must be enrolled in one of the following Levels: Undergraduate "
            "May not be enrolled in one of the following Colleges: College of Arts")
    groups = analysis.restriction_text_groups(text)
    assert {"include": True, "type": "Levels"} in groups
    assert {"include": False, "type": "Colleges"} in groups


def test_restriction_text_groups_handles_classification_and_attribute_phrasings():
    text = ("May not be enrolled as the following Classifications: Freshman "
            "Must be assigned one of the following Student Attributes: Honors")
    types = {(g["include"], g["type"]) for g in analysis.restriction_text_groups(text)}
    assert (False, "Classifications") in types
    assert (True, "Student Attributes") in types


def test_restriction_text_groups_empty():
    assert analysis.restriction_text_groups("") == []
    assert analysis.restriction_text_groups(None) == []


def test_restriction_text_is_selective():
    # "selective" = gated to a specific major/college/program/field, not just level.
    assert analysis.restriction_text_is_selective(
        "May not be enrolled in one of the following Majors: Biology") is True
    assert analysis.restriction_text_is_selective(
        "Must be enrolled in one of the following Levels: Undergraduate") is False


# ── degree-requirement attribute tags (catalog_detail.course_attributes) ─────

def test_parse_attribute_tags_splits_on_comma():
    tags = analysis.parse_attribute_tags("MTH Major_Elective, BUS Minor_Elective")
    assert tags == ["MTH Major_Elective", "BUS Minor_Elective"]
    assert analysis.parse_attribute_tags("") == []


def test_attribute_program_cuts_at_program_keyword():
    assert analysis.attribute_program("MTH Major_Elective") == "MTH Major"
    assert analysis.attribute_program("BUS Minor_Elective") == "BUS Minor"
    # Multiple underscores / degree-track suffixes collapse to the program.
    assert analysis.attribute_program("Economics Major_BAE_Elective") == "Economics Major"


def test_attribute_role():
    assert analysis.attribute_role("MTH Major_Elective") == "Elective"
    assert analysis.attribute_role("COE Major_Required") == "Required"
    assert analysis.attribute_role("Some Gen-Ed Tag") == "Other"
