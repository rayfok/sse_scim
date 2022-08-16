from dataclasses import dataclass
from pathlib import Path
from typing import Union
import json

from mmda.types.document import Document
from pdf2sents.pipeline import Pipeline, PipelineConfig
from pdf2sents.make_output import get_sha_of_pdf
from platformdirs import user_cache_dir

import springs as sp


LOGGER = sp.configure_logging(__file__)


class CachedPipeline(Pipeline):
    def run(self, input_path: Union[Path, str]) -> Document:
        input_path = Path(input_path)
        sha1 = get_sha_of_pdf(input_path)
        cached_loc = Path(user_cache_dir()) / f"{sha1}.json"

        if not cached_loc.exists():
            LOGGER.info(f"No cached version of f{input_path} found,"
                        " running pipeline.")
            with open(cached_loc, 'w') as f:
                data = super().run(input_path)
                json.dump(data.to_json(), f)
        else:
            LOGGER.info(f"Found cached version of {input_path} at "
                        f"{cached_loc}")
            with open(cached_loc, 'r') as f:
                data = Document.from_json(json.load(f))

        return data


@dataclass
class DemandDetailsConfig:
    pipeline: PipelineConfig = PipelineConfig()
    pdf: str = sp.MISSING
    query: str = sp.MISSING





@sp.cli(DemandDetailsConfig)
def main(config: DemandDetailsConfig):
    ...


if __name__ == "__main__":
    main()
