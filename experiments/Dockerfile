FROM ubuntu:18.04

# Environment variables definitions
ENV CONDA_DIR $HOME/miniconda
ENV APP_DIR /app
ENV EXPERIMENTS_DIR $APP_DIR/experiments
ENV LIGHTGBM_DIR $APP_DIR/lightgbm
ENV LIGHTGBM_DIR_PYTHON $LIGHTGBM_DIR/python-package

# Required software installation
RUN apt update && apt-get install wget git build-essential -y
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $CONDA_DIR
ENV PATH=$CONDA_DIR/bin:$PATH

# Creation of new conda enviroment
WORKDIR $APP_DIR
COPY .. $APP_DIR
RUN conda env create -f environment.yml

# Make subsequents RUN commands use the newly created enviroment
SHELL ["conda", "run", "-n", "ilmart", "/bin/bash", "-c"]

# Install the custom LightGBM depedency by hand
RUN git clone --recurse-submodules https://github.com/veneres/LightGBM.git $LIGHTGBM_DIR
WORKDIR $LIGHTGBM_DIR_PYTHON
RUN  python setup.py install

# Return to the initialy app dir
WORKDIR $APP_DIR

# Return to the initialy app dir
RUN  pip install .

WORKDIR $EXPERIMENTS_DIR

ENTRYPOINT ["bin/bash"]