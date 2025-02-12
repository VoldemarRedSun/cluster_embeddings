# NER and cluster entities in medical texts
In this repo, you can make 
* NER of the text
* cluster medical entities 

## Installation
```bash
pip install poetry==1.5.1
git clone https://github.com/VoldemarRedSun/cluster_embeddings.git
cd cluster_embeddings
```
put the file here downloaded from [link](https://drive.google.com/drive/folders/19Ffnh59SIJQPK8XM42ehi_owWP63OGMp?usp=sharing)
```bash
poetry install
poetry shell
python <path_to_script_you_want_to_run>
```
## RUN NER
You can make NER of text typed in cluster_embeddings/spacy_ner.py (for NER your own text you need modify variable text in this py file)
```bash
python cluster_embeddings/spacy_ner.py
```
## RUN cluster entities
File cluster_embeddings/input_data/key_words.txt contains medical key words from articles downlodaded from pubmed database.
File cluster_embeddings/input_data/key_words.h5 contains embeddings of key words made by [biobert](https://github.com/dmis-lab/biobert-pytorch/tree/master/embedding).
Using the command below you can cluster embeddings.
```bash
python cluster_embeddings/spacy_ner.py
```
Output file with cluster and 5 entities per cluster you can find in output_data/key_words_clusters.txt.
All entities with clusters you can find  in output_data/key_words_clusters.json