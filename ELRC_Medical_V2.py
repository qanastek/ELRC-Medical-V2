# coding=utf-8
# Source: https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

"""ELRC-Medical-V2 : European parallel corpus for healthcare machine translation"""

import os
import csv
import datasets
from tqdm import tqdm

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@inproceedings{losch-etal-2018-european,
    title = "European Language Resource Coordination: Collecting Language Resources for Public Sector Multilingual Information Management",
    author = {L{\"o}sch, Andrea  and
      Mapelli, Val{\'e}rie  and
      Piperidis, Stelios  and
      Vasi{\c{l}}jevs, Andrejs  and
      Smal, Lilli  and
      Declerck, Thierry  and
      Schnur, Eileen  and
      Choukri, Khalid  and
      van Genabith, Josef},
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1213",
}
"""

_LANGUAGE_PAIRS = ["en-" + lang for lang in ["bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]]

_LICENSE = """
This work is licensed under a <a rel="license" href="https://elrc-share.eu/static/metashare/licences/CC-BY-4.0.pdf">Attribution 4.0 International (CC BY 4.0) License</a>.
"""

_DESCRIPTION = "No description"

_URLS = {
    "ELRC-Medical-V2": "https://huggingface.co/datasets/qanastek/ELRC-Medical-V2/resolve/main/ELRC_Medical_V2.zip"
}

class ELRC_Medical_V2(datasets.GeneratorBasedBuilder):
    """ELRC-Medical-V2 dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=name, version=datasets.Version("2.0.0"), description="The ELRC-Medical-V2 corpora") for name in _LANGUAGE_PAIRS
    ]

    DEFAULT_CONFIG_NAME = "en-fr"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "lang": datasets.Value("string"),
                "source_text": datasets.Value("string"),
                "target_text": datasets.Value("string"),
            }),
            supervised_keys=None,
            homepage="https://github.com/qanastek/ELRC-Medical-V2/",
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):

        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)

        TRAIN_PATH = 'train.conllu'
        DEV_PATH   = 'dev.conllu'
        TEST_PATH  = 'test.conllu'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, TRAIN_PATH),
                    "split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, DEV_PATH),
                    "split": "dev",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, TEST_PATH),
                    "split": "test",
                }
            ),
        ]
        

    def _generate_examples(self, filepath, split):

        logger.info("⏳ Generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as f:

            guid = 0

            for row in csv.reader(f, delimiter=','):

                print(row)

                yield guid, {
                    "id": str(guid),
                    "lang": "en-fr",
                    "source_text": "hi",
                    "target_text": "salut"
                }

                guid += 1
