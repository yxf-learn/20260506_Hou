from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AblationGrid:
    rows: list[dict] = field(default_factory=list)

    def add(self, name: str, **metrics):
        self.rows.append({"name": name, **metrics})

    def to_records(self):
        return list(self.rows)
