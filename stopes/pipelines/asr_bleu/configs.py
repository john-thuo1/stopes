from dataclasses import dataclass
import typing as tp

@dataclass
class Dataset:
    lang: str
    audio_dirpath: str
    reference_path: str
    reference_format: str
    asr_version: str = "oct22"
    reference_tsv_column: str = None
    audio_format: str = "n_pred.wav"
    results_dirpath: str = None
    transcripts_path: str = None


@dataclass
class CorporaConfig:
    datasets: tp.Dict[str, Dataset]


@dataclass
class AsrBleuConfig:
    launcher: tp.Dict[str, tp.Any]
    corpora: CorporaConfig