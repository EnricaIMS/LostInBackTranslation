# Lost in Back-Translation (WORK IN PROGRESS)

A NMT-based pipeline for the analysis of emotions in back-translations. The pipeline, described in our paper [Lost in Back-Translation:
Emotion Preservation in Neural Machine Translation](add link here), can be used to compare the emotion connotation of input texts vs. their back-translations 
or to perform emotion style transfer via re-ranking of the generated hypotheses. A visualization of the pipeline is shown below.



![procedure](fig/pipeline.png)



## Installation and Requirements
This project runs on python==3.7.3 and uses some additional NLTK packages.

```sh
$ pip install nltk==3.4.4
```
### Translation Module
* [FAIRSEQ](https://fairseq.readthedocs.io/en/latest/)
* [PyTorch](https://pytorch.org) version >= 1.4.0

### Emotion Module
Both the analysis of emotion connotations and the re-ranking of hypotheses are based on an emotion classifier. A different emotion classifiers is used for each emotion corpus.

Put the pretrained emotion classifiers in the folder emotion_module/trained-classifiers.


|    ISEAR | BLOGS | TALES |  TEC  |
|------------:|------------:|-------------:|----------|
|[Download](https://drive.google.com/file/d/1hX0ey3EcVCMdL8ZkQ4Y-YiEmVNT8T_Y2/view?usp=sharing)| [Download](https://drive.google.com/file/d/1gA092woQIeh54omQStThvhLsStLOnH6l/view?usp=sharing) | [Download](https://drive.google.com/file/d/1Oh0V6QQ1dW8j_uqRYwz4FHveUmAkxVPX/view?usp=sharing) | [Download](https://drive.google.com/file/d/1KpfQne8l0QX3sybD3xu6RivUoC-K25eG/view?usp=sharing)|


## Data Preprocessing

0. Clone the repository
```sh 
$ git clone https://github.com/EnricaIMS/LostInBackTranslation.git
$ cd LostInBackTranslation
```
0.1 Get data in folder data



2. Classify the data.

```sh
$ python -m emotion_module.classify_file.py $FILENAME
```

3. Main.py

# Contact
For questions, please contact `enrica.troiano@ims.uni-stuttgart.de`.
