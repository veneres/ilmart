FROM ubuntu:24.04

ARG UID=1001
ARG UNAME=ilmart
RUN useradd -u $UID -rm $UNAME

ENV CODE_HOME=/home/$UNAME/code
ENV LIGHTGBM_DIR=/home/$UNAME/lightgbm
ENV DATA_HOME=/home/$UNAME/data
ENV RANKEVAL_DATA=/home/$UNAME/rankeval_data
ENV TFDS_DATA_DIR=/home/$UNAME/"tfds_data"
ENV QUICKSCORER_PACKAGE=/home/$UNAME/quickscorer_package

ENV DEBIAN_FRONTEND=noninteractive

# Install the required software
RUN apt-get update
RUN apt-get install swig build-essential cmake curl wget git g++-10 gcc-10 numactl libsuitesparse-dev -y

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_24.9.2-0-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR $CODE_HOME
COPY ./environment.yml .
COPY ./pyproject.toml .
COPY ./README.md .
COPY ./src ./src
RUN chown -R $UNAME .

# From here everything is run as the user $UNAME

USER $UNAME

RUN conda init bash

# Create a new conda environment will all the required packages


RUN conda env create -f environment.yml

# Set the environment as the running conda environment
SHELL ["conda", "run", "-n", "ilmart", "/bin/bash", "-c"]

# Ensure to have this pretty old version of numpy to avoid problems during the compilation of rankeval
RUN pip install numpy==1.26.3
RUN pip install rankeval==0.8.2

# Fix tensorflow version
RUN pip uninstall -y tensorflow
RUN pip install tensorflow==2.15.0





WORKDIR $CODE_HOME
# TODO Check why is working only in edit mode
RUN pip install -e .

# Make sure to remove the old version of lightgbm
RUN pip uninstall -y lightgbm



# Install the modified version of lightgbm
RUN git clone --recurse-submodules -b ilmart https://github.com/veneres/LightGBM.git $LIGHTGBM_DIR

WORKDIR $LIGHTGBM_DIR
RUN sh ./build-python.sh install

# Make sure to have the fast version (without the python binding) of ligthgbm installed
RUN mkdir build
WORKDIR $LIGHTGBM_DIR/build
RUN cmake ..
RUN make -j4


# Build the fast_distilled version of Ilmart
WORKDIR $CODE_HOME/src/ilmart/fast_distilled
RUN mkdir build
WORKDIR $CODE_HOME/src/ilmart/fast_distilled/build
    RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make




# Make the environment as the default environment when running the container
RUN echo "conda activate ilmart" >> ~/.bashrc




# Set a standard set of environment variables

WORKDIR /home/$UNAME/

# Folder that will be used as a volume to store the data and communite with the host
RUN mkdir $DATA_HOME
RUN mkdir $RANKEVAL_DATA
RUN mkdir $QUICKSCORER_PACKAGE
RUN mkdir $TFDS_DATA_DIR

ENTRYPOINT ["bash"]