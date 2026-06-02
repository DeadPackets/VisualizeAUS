"""Pure helpers for analysing AUSCrawl's structured JSON columns.

AUSCrawl stores three rich JSON structures that the dashboard turns into
metrics:

* ``section_details.prerequisites_json`` / ``corequisites_json`` — a boolean
  expression *tree* of AND/OR groups over course leaves.
* ``section_details.restrictions_json`` — a list of typed include/exclude
  enrolment groups.
* ``catalog_detail.course_attributes`` — comma-separated degree-requirement
  tags such as ``"MTH Major_Elective"``.

Everything here is pure and import-safe so it can be unit-tested without
running the (procedural) site generator in ``build.py``.
"""

import json
import re

# ── prerequisite / corequisite expression trees ─────────────────────────────


def load_tree(raw):
    """Parse a ``*_json`` string into a tree, or ``None`` if it is empty/junk.

    Empty strings, ``"[]"``, ``"null"`` and malformed JSON all collapse to
    ``None`` so callers can simply test truthiness.
    """
    if not raw:
        return None
    try:
        tree = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not tree or not isinstance(tree, dict):
        return None
    return tree


def _operands(node):
    return node.get("operands", []) if isinstance(node, dict) else []


def tree_courses(node):
    """Return every course-leaf dict in the tree, in document order."""
    if not isinstance(node, dict):
        return []
    if node.get("type") == "course":
        return [node]
    out = []
    for child in _operands(node):
        out.extend(tree_courses(child))
    return out


def tree_has_or(node):
    """True if any OR group appears anywhere in the tree."""
    if not isinstance(node, dict):
        return False
    if node.get("type") == "or":
        return True
    return any(tree_has_or(c) for c in _operands(node))


def tree_depth(node):
    """Operator-nesting depth: a bare course is 1, a flat AND/OR is 2, etc."""
    kids = _operands(node)
    if not kids:
        return 1
    return 1 + max(tree_depth(c) for c in kids)


def tree_mandatory_count(node):
    """Number of courses that *must* be taken (leaves not inside any OR)."""
    if not isinstance(node, dict):
        return 0
    t = node.get("type")
    if t == "course":
        return 1
    if t == "or":
        return 0
    return sum(tree_mandatory_count(c) for c in _operands(node))


def tree_alternative_paths(node):
    """Distinct ways to satisfy the requirement (AND multiplies, OR adds)."""
    if not isinstance(node, dict):
        return 1
    t = node.get("type")
    if t == "course":
        return 1
    if t == "or":
        return sum(tree_alternative_paths(c) for c in _operands(node))
    product = 1
    for c in _operands(node):
        product *= tree_alternative_paths(c)
    return product


def tree_has_concurrent(node):
    """True if any leaf may be taken concurrently ('may be taken together')."""
    return any(c.get("concurrent") for c in tree_courses(node))


def classify_prereq_shape(node):
    """Bucket a tree into a human-readable structural category."""
    if not isinstance(node, dict):
        return "None"
    if node.get("type") == "course":
        return "Single course"
    if tree_has_or(node):
        return "Complex (nested AND/OR)" if tree_depth(node) >= 3 else "Has alternatives (OR)"
    return "All required (AND)"


# ── enrolment restrictions ──────────────────────────────────────────────────

_RESTRICTION_TYPE_LABELS = {
    "Levels": "Level",
    "Classifications": "Class standing",
    "Majors": "Major",
    "Colleges": "College",
    "Programs": "Program",
    "Campus": "Campus",
    "Cohort": "Cohort",
    "Degrees": "Degree",
    "Student Attributes": "Student attribute",
    "Fields of Study (Major, Minor, or Concentration)": "Field of study",
}


def restriction_groups(raw):
    """Parse ``restrictions_json`` into a list of group dicts (``[]`` if junk)."""
    if not raw:
        return []
    try:
        groups = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return groups if isinstance(groups, list) else []


def normalize_restriction_type(t):
    """Shorten Banner's restriction-type names for chart labels."""
    return _RESTRICTION_TYPE_LABELS.get(t, str(t).rstrip("s"))


def restriction_label(group):
    """A ``"Must be: Major"`` / ``"Must not be: College"`` style label."""
    verb = "Must be" if group.get("include") else "Must not be"
    return f"{verb}: {normalize_restriction_type(group.get('type'))}"


# Banner phrases restriction rules very regularly, e.g.
#   "Must be enrolled in one of the following Levels: Undergraduate"
#   "May not be enrolled as the following Classifications: Freshman"
#   "Must be assigned one of the following Student Attributes: Honors"
# We only need the verb (include/exclude) and the type noun before the colon.
_RESTRICTION_TEXT_RE = re.compile(
    r"(Must|May not) be [a-z ]*?following ([A-Za-z][A-Za-z ()/,]*?):")

_SELECTIVE_TYPES = {"Majors", "Colleges", "Programs",
                    "Fields of Study (Major, Minor, or Concentration)"}


def restriction_text_groups(text):
    """Parse the raw ``restrictions`` text into typed include/exclude groups.

    Used instead of ``restrictions_json`` because the JSON is only populated
    for recently-fetched sections, while this text is present for ~90% of them.
    """
    if not text:
        return []
    out = []
    for verb, rtype in _RESTRICTION_TEXT_RE.findall(str(text)):
        out.append({"include": verb == "Must", "type": rtype.strip()})
    return out


def restriction_text_is_selective(text):
    """True if the text gates to a specific major/college/program/field
    (not merely an academic level)."""
    return any(g["type"] in _SELECTIVE_TYPES for g in restriction_text_groups(text))


# ── degree-requirement attribute tags ───────────────────────────────────────

_PROGRAM_KEYWORDS = ("Major", "Minor", "Concentration", "Certificate")


def parse_attribute_tags(raw):
    """Split a ``course_attributes`` string into individual tags."""
    if not raw:
        return []
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def attribute_program(tag):
    """The program a tag belongs to, e.g. ``"Economics Major_BAE_Elective"`` →
    ``"Economics Major"``."""
    for kw in _PROGRAM_KEYWORDS:
        idx = tag.find(kw)
        if idx != -1:
            return tag[: idx + len(kw)].strip()
    return tag.split("_")[0].strip()


def attribute_role(tag):
    """Whether a tag marks the course as an Elective, Required, Core, or Other."""
    low = tag.lower()
    if "elective" in low:
        return "Elective"
    if "required" in low:
        return "Required"
    if "core" in low:
        return "Core"
    return "Other"
