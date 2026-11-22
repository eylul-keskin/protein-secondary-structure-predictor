from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectPaths:
    """Centralizes dataset and output paths."""

    root: Path
    train_csv: Path = field(init=False)
    validation_csv: Path = field(init=False)
    cb513_csv: Path = field(init=False)
    ts115_csv: Path = field(init=False)
    casp12_csv: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    models_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.train_csv = self.root / "training_secondary_structure_train.csv"
        self.validation_csv = self.root / "validation_secondary_structure_valid.csv"
        self.cb513_csv = self.root / "test_secondary_structure_cb513.csv"
        self.ts115_csv = self.root / "test_secondary_structure_ts115.csv"
        self.casp12_csv = self.root / "test_secondary_structure_casp12.csv"
        self.outputs_dir = self.root / "outputs"
        self.figures_dir = self.outputs_dir / "figures"
        self.metrics_dir = self.outputs_dir / "metrics"
        self.predictions_dir = self.outputs_dir / "predictions"
        self.logs_dir = self.outputs_dir / "logs"
        self.models_dir = self.outputs_dir / "models"

    @property
    def split_files(self) -> dict[str, Path]:
        return {
            "train": self.train_csv,
            "validation": self.validation_csv,
            "cb513": self.cb513_csv,
            "ts115": self.ts115_csv,
            "casp12": self.casp12_csv,
        }

    @property
    def output_dirs(self) -> list[Path]:
        return [
            self.outputs_dir,
            self.figures_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.logs_dir,
            self.models_dir,
        ]


@dataclass
class TrainingConfig:
    """Reasonable defaults for a course-project BiLSTM."""

    seed: int = 42
    batch_size: int = 16
    epochs: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    gradient_clip: float = 1.0
    patience: int = 3
    ignore_index: int = -100
    num_workers: int = 0
    length_bucket_edges: tuple[int, int] = (100, 300)
    sample_prediction_count: int = 8


@dataclass
class ProjectConfig:
    """Top-level configuration used by the scripts."""

    paths: ProjectPaths
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sequence_column: str = "seq"
    target_column: str = "sst8"
    auxiliary_target_column: str = "sst3"
    preferred_device: str = "auto"


def build_config(root: Path) -> ProjectConfig:
    """Builds the default configuration for this project root."""

    return ProjectConfig(paths=ProjectPaths(root=root))
