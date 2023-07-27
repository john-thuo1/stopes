from dataclasses import dataclass, field

@dataclass
class AsrBleuConfig:
    lang: str
    audio_dirpath: str
    reference_path: str
    reference_format: str
    # launcher: tp.Dict[str, tp.Any]
    asr_version: str = "oct22"
    reference_tsv_column: str = None
    audio_format: str = "n_pred.wav"
    results_dirpath: str = None
    transcripts_path: str = None