#FROM alpine:3.1
FROM continuumio/miniconda3

ADD . /
#Creating an environment

RUN conda env create -f environment.yml


# Pull the environment name out of the environment.yml
RUN echo "source activate py36" > ~/.bashrc
ENV PATH /opt/conda/envs/py36/bin:$PATH
# Bundle app 
EXPOSE  80
CMD ["python", "api_predict.py"]
