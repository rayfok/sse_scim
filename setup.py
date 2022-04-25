
import setuptools

setuptools.setup(
    name="mmda_pdf_scorer",
    version="0.0.2",
    python_requires=">= 3.8",
    packages=setuptools.find_packages(),
    setup_requires=[
        'torch==1.9.0'
    ],
    install_requires=[
        'espresso-config==0.10.0',
        'spacy==3.2.4',
        'thefuzz[speedup]==0.19.0',
        'pylcs==0.0.6',
        'pydantic==1.8.2',
        'textacy==0.12.0',
        'nltk==3.7.0',
        'pdfplumber==0.6.0',
        'pandas==1.4.1',
        'requests==2.26.0',
        'torch==1.9.0',
        'torchvision==0.10.0',
        'PyPDF2==1.26.0',
        'transformers==4.5',
        'layoutparser[effdet]==0.3.4',
        'vila[lp_predictors,vila_predictors]==0.3.0',
        'detectron2 @ git+https://github.com/facebookresearch/detectron2@v0.4',
        'mmda @ git+https://github.com/allenai/mmda@lucas/fix-float-bboxes'
    ],
)
