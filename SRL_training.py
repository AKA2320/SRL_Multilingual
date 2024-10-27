# %% [markdown]
# # env

# %%
!which python
!python --version

%env PYTHONPATH=

!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local

!conda create -n SRL-S2S python=3.6 -y

!conda init

# %%
!source activate SRL-S2S && pip install allennlp==0.8.4 spacy==2.1.0 torch==1.1.0
!source activate SRL-S2S && pip install flair==0.4.3 botocore==1.18.18 awscli==1.18.159 boto3==1.15.18
!git clone https://github.com/Heidelberg-NLP/SRL-S2S.git
!source activate SRL-S2S && pip install overrides==3.1.0 scikit-learn==0.22.2 numpy==1.19.1 torch==1.1.0

# %% [markdown]
# # code

# %%
# upload data to env
# make changes to data path and embedding path in config
# german embeddings: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz
# upload all src files to runtime
# reduce epochs if necessary

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
!source activate SRL-S2S && allennlp train /content/SRL-S2S/training_config/monolingual/de_copynet-srl-conll09.json \
-s /content/drive/MyDrive/de_actual_new11/ \
--include-package seq2seq_copy_srl_reader \
--include-package seq2seq_copy_srl_predictor \
--include-package seq2seq_copynet_srl

# %%
!source activate SRL-S2S && allennlp predict /content/drive/MyDrive/de_actual_new11/model.tar.gz /content/ENtoDE_test.json \
	--output-file /content/drive/MyDrive/de_actual_new11/predopt.json \
	--include-package seq2seq_copy_srl_reader \
  --include-package seq2seq_copy_srl_predictor \
  --include-package seq2seq_copynet_srl \
	--predictor seq2seq-srl

# %%



