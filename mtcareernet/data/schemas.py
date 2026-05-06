"""Frozen column dictionaries — paired with the data manifest."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    nullable: bool = False
    description: str = ""


RECORD_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("student_id", "int64"),
    FieldSpec("cohort_year", "int16"),
    FieldSpec("institution_code", "category"),
    FieldSpec("major_code", "category"),
    FieldSpec("gender", "category"),
    FieldSpec("age", "int8"),
    FieldSpec("rural_origin", "int8"),
    FieldSpec("first_generation", "int8"),
    FieldSpec("ethnicity", "category"),
    FieldSpec("gpa_s1", "float32", nullable=True),
    FieldSpec("gpa_s2", "float32", nullable=True),
    FieldSpec("gpa_s3", "float32", nullable=True),
    FieldSpec("gpa_s4", "float32", nullable=True),
    FieldSpec("gpa_s5", "float32", nullable=True),
    FieldSpec("gpa_s6", "float32", nullable=True),
    FieldSpec("gpa_s7", "float32", nullable=True),
    FieldSpec("gpa_s8", "float32", nullable=True),
    FieldSpec("internship_count", "int8"),
    FieldSpec("club_membership", "int8"),
    FieldSpec("leadership_indicator", "int8"),
    FieldSpec("cv_text", "string", nullable=True),
)

OUTCOME_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("y_career_readiness", "float32", nullable=True),
    FieldSpec("y_employment_6m", "int8", nullable=True),
    FieldSpec("y_major_job_alignment", "int8", nullable=True),
    FieldSpec("y_log_starting_salary", "float32", nullable=True),
)
