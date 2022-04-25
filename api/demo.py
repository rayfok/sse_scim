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

from app.parser import MmdaPdfParser, MmdaParserConfig

LOGGER = configure_logging(logger_name=__file__, logging_level=logging.INFO)


class Config(ConfigNode):
    class create(ConfigNode):
        live: ConfigParam(bool) = False

    class launch(ConfigNode):
        server_name: ConfigParam(str) = "0.0.0.0"
        server_port: ConfigParam(int) = 3000
        enable_queue: ConfigParam(bool) = True



@cli(Config, print_fn=logging.warn)
def main(config: Config):
    gr.close_all()

    pdf_parser = MmdaPdfParser()

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
