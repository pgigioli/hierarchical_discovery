FROM tensorflow/tensorflow:1.11.0-devel-gpu-py3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV SHELL=/bin/bash

# Install packages
RUN apt-get update && apt-get install -y \
    nano \
    htop

# Upgrade pip
RUN pip install --upgrade pip

# Create working directory
RUN mkdir /hierarchical_discovery

# Install custom dependencies
ADD requirements.txt /hierarchical_discovery
WORKDIR /hierarchical_discovery
RUN pip install -r requirements.txt

# Download nltk stopwords corpus
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader reuters

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--no-browser"]
