import logging
from aiofiles.tempfile import NamedTemporaryFile as AsyncNamedTemporaryFile
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from pathlib import Path


LOGGER = logging.getLogger('uvicorn')


def save_temp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp:
            tmp.write(upload_file.read())
            tmp_path = Path(tmp.name)

        LOGGER.debug(f'Uploaded to {tmp_path}')

    finally:
        upload_file.file.close()
    return tmp_path


async def async_save_temp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix

        async with AsyncNamedTemporaryFile(delete=False,
                                           suffix=suffix,
                                           mode='wb') as tmp:
            content = True
            while content:
                content = await upload_file.read(1024)
                await tmp.write(content)
            tmp_path = Path(tmp.name)

        LOGGER.debug(f'Uploaded to {tmp_path}')

    finally:
        upload_file.file.close()
    return tmp_path
