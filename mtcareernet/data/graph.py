"""Heterogeneous student-major-occupation-skill graph."""
from __future__ import annotations
from dataclasses import dataclass, field
import torch


REL_TYPES = ("enrolled_in", "trains", "requires", "employed_in")


@dataclass
class HeteroGraph:
    n_students: int
    n_majors: int
    n_occupations: int
    n_skills: int
    adjacency: dict[str, torch.Tensor] = field(default_factory=dict)
    node_features: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.n_students + self.n_majors + self.n_occupations + self.n_skills


def build_smos_graph(student_major, major_skill, occ_skill, student_occ_past,
                     n_students: int, n_majors: int, n_occupations: int, n_skills: int) -> HeteroGraph:
    n_total = n_students + n_majors + n_occupations + n_skills
    A = {r: torch.zeros(n_total, n_total) for r in REL_TYPES}
    s0, m0, o0, k0 = 0, n_students, n_students + n_majors, n_students + n_majors + n_occupations
    for s, m in student_major:
        A["enrolled_in"][s0 + s, m0 + m] = 1
    for m, k in major_skill:
        A["trains"][m0 + m, k0 + k] = 1
    for o, k in occ_skill:
        A["requires"][o0 + o, k0 + k] = 1
    for s, o in student_occ_past:
        A["employed_in"][s0 + s, o0 + o] = 1
    return HeteroGraph(n_students, n_majors, n_occupations, n_skills, adjacency=A)
