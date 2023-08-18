import json
import logging
from pathlib import Path
import typing as tp
import fairseq


import torch
from tqdm import tqdm

from stopes.core import utils

try:
    import torchaudio
    from torchaudio.models.decoder import ctc_decoder
except ImportError:
    raise ImportError("Upgrade torchaudio to 0.12 to enable CTC decoding")

logger = logging.getLogger("stopes.asr_bleu.utils")


class DownloadProgressBar(tqdm):
    """A class to represent a download progress bar"""

    def update_to(self, b=1, bsize=1, tsize=None) -> None:
        """
        Update the download progress
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def retrieve_asr_config(
    lang_key: str,
    asr_version: str,
    json_path: str
) -> tp.Dict:
    if len(lang_key) != 3:
        raise ValueError(
            f"'{lang_key}' lang key for language type must be 3 characters!"
        )

    with open(json_path, "r") as f:
        asr_model_cfgs = json.load(f)
    return asr_model_cfgs[lang_key][asr_version]


class ASRContainer(object):
    """A class to represent a ASR generator"""

    def __init__(
        self,
        model_cfg: tp.Dict,
        cache_dirpath: str = (Path.home() / ".cache" / "ust_asr").as_posix(),
    ) -> None:
        """
        Construct all the necessary attributes of the ASRGenerator class

        Args:
            model_cfg: the dict of the asr model config
            cache_dirpath: the default cache path is
            "Path.home()/.cache/ust_asr"
        """

        self.cache_dirpath = Path(cache_dirpath) / model_cfg["lang"]
        self.model_cfg = model_cfg

        self.use_cuda = torch.cuda.is_available()

        torchaudio.set_audio_backend("sox_io")

        if self.model_cfg["model_type"] == "hf":
            self.prepare_hf_model(self.model_cfg)
        elif self.model_cfg["model_type"] == "fairseq":
            self.prepare_fairseq_model(self.model_cfg)
        else:
            raise NotImplementedError(
                f"Model type {self.model_cfg['model_type']} is not supported"
            )

        if self.model_cfg["post_process"] == "collapse":
            self.post_process_fn = lambda hypo: "".join(hypo).replace(
                self.sil_token, " "
            )
        elif self.model_cfg["post_process"] == "none":
            self.post_process_fn = lambda hypo: " ".join(hypo).replace(
                self.sil_token, " "
            )
        else:
            raise NotImplementedError

        if self.use_cuda:
            self.model.cuda()
        self.model.eval()

        self.decoder = ctc_decoder(
            lexicon=None,
            tokens=self.tokens,
            lm=None,
            nbest=1,
            beam_size=1,
            beam_size_token=None,
            lm_weight=0.0,
            word_score=0.0,
            unk_score=float("-inf"),
            sil_token=self.sil_token,
            sil_score=0.0,
            log_add=False,
            blank_token=self.blank_token,
        )

    def prepare_hf_model(self, model_cfg: tp.Dict) -> None:
        """
        Prepare the huggingface asr model

        Args:
            model_cfg: dict with the relevant ASR config
        """

        def infer_silence_token(vocab: list):
            """
            Different HF checkpoints have different notion of silence token
            such as | or " " (space)
            Important: when adding new HF asr model in,
            check what silence token it uses
            """
            if "|" in vocab:
                return "|"
            elif " " in vocab:
                return " "
            else:
                raise RuntimeError(
                    "Silence token is not found in the vocabulary"
                )
        try:
            from transformers import (AutoFeatureExtractor, AutoTokenizer,
                                      Wav2Vec2ForCTC, Wav2Vec2Processor)
        except ImportError:
            raise ImportError("Install transformers to load HF wav2vec model")

        model_path = model_cfg["model_path"]
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.preprocessor = AutoFeatureExtractor.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)

        # extra unk tokens are there to make some models work
        # e.g. Finnish ASR has some vocab issue
        vocab_list = [
            self.tokenizer.decoder.get(i, f"{self.tokenizer.unk_token}1")
            for i in range(self.tokenizer.vocab_size)
        ]
        self.sampling_rate = self.preprocessor.sampling_rate
        self.normalize_input = self.preprocessor.do_normalize
        self.tokens = vocab_list
        self.sil_token = infer_silence_token(vocab_list)
        self.blank_token = self.tokenizer.pad_token

    def prepare_fairseq_model(self, model_cfg: tp.Dict) -> None:
        ckpt_path = model_cfg["ckpt_path"]
        dict_path = model_cfg["dict_path"]
        lang_post = model_cfg["post_process"]

        logger.info(f"language post process : {lang_post}")
        logger.info(f"Checkpoint path : {ckpt_path}")
        logger.info(f"Dictionary  path : {dict_path}")
        model, saved_cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [ckpt_path],
                arg_overrides={
                    "task": "audio_finetuning",
                    "data": self.cache_dirpath.as_posix(),
                },  # data must have dict in it
            )
        
        logger.info(f"Saved_cfg path : {saved_cfg}")
        dict_lines = utils.open(dict_path, "r").readlines()
        tokens = [line.split()[0] for line in dict_lines]
        # adding default fairseq special tokens
        tokens = ["<s>", "<pad>", "</s>", "<unk>"] + tokens

        self.model = model[0]
        self.tokens = tokens

        if "|" in tokens:
            self.sil_token = "|"
        else:
            self.sil_token = tokens[2]  # use eos as silence token if | not presented e.g. Hok ASR model
        logger.info(f"Inferring silence token from the dict: {self.sil_token}")
        self.blank_token = self.tokens[0]

        self.sampling_rate = saved_cfg.task.sample_rate
        self.normalize_input = saved_cfg.task.normalize
