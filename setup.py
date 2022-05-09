
import setuptools

import subprocess
import sys
import pathlib


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# We must install torch here cuz detectron2 requires it during setup
install('torch==1.9.0')

with open(pathlib.Path(__file__).parent() / 'requirements.txt') as f:
    requirements = [ln.strip() for ln in f]

setuptools.setup(
    name="mmda_pdf_scorer",
    version="0.0.4",
    python_requires=">= 3.8",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    install_requires=requirements,
)
