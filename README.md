# TrAnET: Tracking and Analyzing the Evolution of Topics in Information Networks

### Dataset

The dataset is available in the `dataset` folder and is encoded in the ArnetMiner v8 format.

To unzip 

	cat merged-dataset-v8-splitted.z* > merged-dataset-v8-splitted-complete.zip
	unzip merged-dataset-v8-splitted-complete.zip

### Installation

Install dependencies 

	pip install -r requirements.txt

Download the nltk data: 

a) open a python shell

    python
    
b) import nktk and download lemmer data

    import nltk
    nltk.download('wordnet')
    nltk.download('punkt')

#### For ubuntu users

Install python3-dev:

	sudo apt-get install python3-dev

    
### Run the demo

1) Complete the `config.py` file with requested information

2) Ingest the dataset executing `data_ingestion.main()`

3) Run the notebook `jupyter notebook notebook`

