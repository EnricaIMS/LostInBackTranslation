# Lost in Back-Translation

A NMT-based pipeline for the analysis of emotions in back-translations. The pipeline, described in our paper [Lost in Back-Translation:
Emotion Preservation in Neural Machine Translation](add link here), can be used to compare the emotion connotation of input texts vs. their back-translations 
or to perform emotion style transfer via re-ranking of the generated hypotheses. A visualization of the pipeline is shown below.

![procedure](fig/pipeline.png)

## Installation
This project runs on python==3.7.3. It is requires on [FAIRSEQ](https://fairseq.readthedocs.io/en/latest/) and [PyTorch](https://pytorch.org) version >= 1.4.0. Install additional dependencies with:

```sh
$ pip install nltk==3.4.4
```

### Other Requirements
Both the analysis of emotion connotations and the re-ranking of hypotheses are based on an emotion classifier. You are free to use one of your choice.

Put ...
