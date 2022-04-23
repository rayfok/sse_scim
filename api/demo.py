import logging
from tempfile import NamedTemporaryFile
import gradio as gr
from espresso_config import (
    ConfigNode,
    ConfigParam,
    cli,
    config_to_dict,
    configure_logging
)

from app.parser import parse_pdf, MmdaParserConfig

LOGGER = configure_logging(logger_name=__file__, logging_level=logging.INFO)


class Config(ConfigNode):
    class create(ConfigNode):
        live: ConfigParam(bool) = False

    class launch(ConfigNode):
        server_name: ConfigParam(str) = "0.0.0.0"
        server_port: ConfigParam(int) = 3000
        enable_queue: ConfigParam(bool) = True


class PdfParser():
    def __init__(self):
        self.config = MmdaParserConfig()

    def parse_pdf(self, temp_fn: NamedTemporaryFile):
        parsed = parse_pdf(temp_fn.name, config=self.config)
        return [s.to_json() for s in parsed]



@cli(Config, print_fn=logging.warn)
def main(config: Config):
    gr.close_all()

    pdf_parser = PdfParser()

    interface = gr.Interface(
        fn=pdf_parser.parse_pdf,
        inputs=[
            gr.inputs.File(file_count="single", type="file", label='PDF')
        ],
        outputs=[
            gr.outputs.JSON(label='Parse Output')
        ],
        **config_to_dict(config.create)
    )

    try:
        interface.launch(**config_to_dict(config.launch))
    except KeyboardInterrupt:
        print('\nBye!')
        interface.close()


if __name__ == '__main__':
    main()
