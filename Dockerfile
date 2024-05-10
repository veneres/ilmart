FROM continuumio/miniconda3:23.10.0-1

ARG UID=1000
ARG UNAME=testuser
RUN useradd -u $UID -m $UNAME


ENV CODE_HOME /code
ENV LIGHTGBM_DIR /home/$UNAME/lightgbm

ENV DEBIAN_FRONTEND=noninteractive

# Install the required software
RUN apt-get update
RUN apt-get install default-jdk swig build-essential cmake -y


WORKDIR $CODE_HOME
COPY ./environment.yml .
COPY ./setup.cfg .
COPY ./pyproject.toml .
COPY ./README.md .
COPY ./src ./src
RUN chown -R $UNAME .


# Create dumb user just in case or use the value from the command line
# From here everything is run as the user


USER $UNAME

RUN conda init bash

# Create a new conda environment will all the required packages


RUN conda env create -f environment.yml

# Set the environment as the running conda environment
SHELL ["conda", "run", "-n", "ilmart", "/bin/bash", "-c"]

# Install the modified version of lightgbm
RUN git clone --recurse-submodules -b ilmart https://github.com/veneres/LightGBM.git $LIGHTGBM_DIR
WORKDIR $LIGHTGBM_DIR
RUN sh ./build-python.sh install
RUN mkdir build
WORKDIR $LIGHTGBM_DIR/build
RUN cmake ..
RUN make -j4

RUN pip install rankeval==0.8.2


# Fix tensorflow version
RUN pip uninstall -y tensorflow
RUN pip install tensorflow==2.15.0


WORKDIR $CODE_HOME
# TODO Check why is working only in edit mode
RUN pip install -e .

#RUN python -c 'import ilmart; print(ilmart.__version__)'

# Make the environment as the default environment when running the container
RUN echo "conda activate ilmart" >> ~/.bashrc



# Set a standard set of environment variables
ENV IR_DATASETS_HOME="/ir_datasets"
ENV TRANSFORMERS_CACHE="/hf"
ENV RANKEVAL_DATA="/rankeval_data"
ENV TFDS_DATA_DIR="/tfds_data"
WORKDIR $CODE_HOME