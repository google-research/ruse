FROM gcr.io/tpu-pytorch/xla:r1.7

RUN mkdir /workdir
WORKDIR /workdir

ENV TFDS_DATA_DIR=gs://tensorflow-datasets/datasets
RUN apt-get update && apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip
COPY seq2seq/requirements.txt /workdir/seq2seq/requirements.txt
RUN python3 -m pip --no-cache-dir install --use-feature=2020-resolver -r seq2seq/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
COPY seq2seq /workdir/seq2seq

#COPY entrypoint.sh /workdir/entrypoint.sh
#RUN chmod +x ./entrypoint.sh
#ENTRYPOINT ["./entrypoint.sh"]
