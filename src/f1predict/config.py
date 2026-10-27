from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root() -> Path:
    """Find the repository root without depending on the caller's working directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def default(cls) -> DataPaths:
        return cls(find_project_root() / "data")

    @property
    def raw_jolpica(self) -> Path:
        return self.root / "raw" / "jolpica"

    @property
    def raw_f1db(self) -> Path:
        return self.root / "raw" / "f1db"

    @property
    def raw_openf1(self) -> Path:
        return self.root / "raw" / "openf1"

    @property
    def fastf1_cache(self) -> Path:
        return self.root / "cache" / "fastf1"

    @property
    def processed_jolpica(self) -> Path:
        return self.root / "processed" / "jolpica"

    @property
    def processed_f1db(self) -> Path:
        return self.root / "processed" / "f1db"

    @property
    def processed_openf1(self) -> Path:
        return self.root / "processed" / "openf1"

    @property
    def processed_fastf1(self) -> Path:
        return self.root / "processed" / "fastf1"

    @property
    def catalog(self) -> Path:
        return self.root / "catalog.json"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    def create(self) -> None:
        for path in (
            self.raw_jolpica,
            self.raw_f1db,
            self.raw_openf1,
            self.fastf1_cache,
            self.processed_jolpica,
            self.processed_f1db,
            self.processed_openf1,
            self.processed_fastf1,
        ):
            path.mkdir(parents=True, exist_ok=True)
