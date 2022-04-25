FROM soldni/mmda-pytorch-transformer-cpu:1.0

# Setup a spot for the api code
WORKDIR /api

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install -r requirements.txt

########## COPYING SOURCE CODE FROM HERE ON ##########

# Copy over the source code
COPY app app/
COPY demo.py demo.py
COPY serve.py serve.py

# install the spacy model
RUN python -m spacy download en_core_web_sm

# Kick things off
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0"]