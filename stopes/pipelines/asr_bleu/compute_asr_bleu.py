import asyncio
import logging
import sacrebleu
from stopes.pipelines.asr_bleu.configs import AsrBleuConfig
from tqdm import tqdm
from stopes.pipelines.asr_bleu.asr_generator import ASRGenerator
from stopes.pipelines.asr_bleu.transcribe_audio import transcribe_audiofile
from stopes.pipelines.asr_bleu.retrieve_data import (retrieve_data, retrieve_asr_config)                                                
import hydra


logger = logging.getLogger("asr_bleu")


class AsrBleu:
    def __init__(self, config: AsrBleuConfig):
        self.config = config
        self.launcher = hydra.utils.instantiate(self.config.launcher)

    async def run(self):
        # 1. Retrieve ASR configuration 

        asr_config = retrieve_asr_config(self.config.corpora.lang, self.config.corpora.asr_version, json_path="/home/john/Desktop/STOPES/stopes/stopes/pipelines/asr_bleu/conf/asr_models/asr_model_cfgs.json")
        asr_model = ASRGenerator(asr_config)
        
        # 2. Evaluation data

        eval_manifest = await retrieve_data(
              [(self.config.corpora.audio_dirpath, self.config.corpora.reference_path)], 
                self.launcher,
                self.config.corpora.audio_format,
                self.config.corpora.reference_format,
                self.config.corpora.reference_tsv_column
        )

        # 3. Transcribe audio predictions and compute BLEU score.
        prediction_transcripts = []
        for _, eval_pair in tqdm(
            eval_manifest.iterrows(),
            desc="Transcribing predictions",
            total=len(eval_manifest),
        ):
            transcription = await transcribe_audiofile(asr_model,eval_pair.prediction)
            prediction_transcripts.append(transcription.lower())

        if self.config.corpora.lang == "hok":
            prediction_transcripts = [
                merge_tailo_init_final(text) for text in prediction_transcripts
            ]

        references = eval_manifest["reference"].tolist()
        bleu_score = sacrebleu.corpus_bleu(prediction_transcripts, [references])

        print(bleu_score)

        
        return prediction_transcripts, bleu_score   

def merge_tailo_init_final(text):
    """
    Hokkien ASR hypothesis post-processing.
    """
    sps = text.strip().split()
    results = []
    last_syllable = ""
    for sp in sps:
        if sp == "NULLINIT" or sp == "nullinit":
            continue
        last_syllable += sp
        if sp[-1].isnumeric():
            results.append(last_syllable)
            last_syllable = ""
    if last_syllable != "":
        results.append(last_syllable)
    return " ".join(results)


def remove_tone(text):
    """
    Used for tone-less evaluation of Hokkien
    """
    return " ".join([t[:-1] for t in text.split()])
 

@hydra.main(config_path="conf", config_name="asr_bleu")
def main(config: AsrBleuConfig) -> None:
    pipeline = AsrBleu(config)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()        
