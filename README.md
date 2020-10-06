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
Both the analysis of emotion connotations and the re-ranking of hypotheses are based on an emotion classifier. You are free to use one of your choice.

Put ...

## Data Preprocessing

0. Clone the repository
```sh 
$ git clone https://github.com/EnricaIMS/LostInBackTranslation.git
$ cd LostInBackTranslation
```

1. Preprocess the data.

```sh
$ python -m sesame.preprocess
```
The code above puts the data in the right format (Sentence_id EmotionLabel TargetEmotion Text).
The same format is required both for analysis, emotion recover and transfer.

# Contact
For questions, please contact `enrica.troiano@ims.uni-stuttgart.de`.
