# Lost in Back-Translation (WORK IN PROGRESS)

A NMT-based pipeline for the analysis of emotions in back-translations. The pipeline, described in our paper [Lost in Back-Translation:
Emotion Preservation in Neural Machine Translation](add link here), can be used to compare the emotion connotation of input texts vs. their back-translations 
or to perform emotion style transfer via re-ranking of the generated hypotheses. A visualization of the pipeline is shown below.

<p align="center">
<img align="center" src="fig/pipeline.png" width="790" height="120">
</p>

## Installation and Requirements
This project runs on [Python](https://www.python.org) version >= 3.7.0. Clone the repository and install the additional packages with:

```sh
$ git clone https://github.com/EnricaIMS/LostInBackTranslation.git
$ cd LostInBackTranslation
$ pip install -r Requirements.txt
```

* **Translation Module:** Download the [ensamble models](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md) pretrained by [Ng et al.(2019)](https://www.aclweb.org/anthology/W19-5333.pdf) and store them in scripts/translation_module/translation_models.


* **Emotion Modules:** Both the emotion-informed selection and the subsequent analysis of emotion change are based on an emotion classifier. Download and move our pretrained BiLSTMs in the folder emotion_module/trained-classifiers.

|[ISEAR](https://drive.google.com/file/d/1hX0ey3EcVCMdL8ZkQ4Y-YiEmVNT8T_Y2/view?usp=sharing)| [BLOGS](https://drive.google.com/file/d/1gA092woQIeh54omQStThvhLsStLOnH6l/view?usp=sharing) | [TALES](https://drive.google.com/file/d/1Oh0V6QQ1dW8j_uqRYwz4FHveUmAkxVPX/view?usp=sharing) | [TEC](https://drive.google.com/file/d/1KpfQne8l0QX3sybD3xu6RivUoC-K25eG/view?usp=sharing)|
|------------|------------|------------|------------|

### Input Data
A different classifiers is used for the data coming from a specific emotion corpus:
### Examples



# Contact
For questions, please contact `enrica.troiano@ims.uni-stuttgart.de`.
