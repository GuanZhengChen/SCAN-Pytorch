## SCAN: Semi-supervisedly Co-embedding Attributed Networks
This repository contains the Python&Pytorch implementation for SCAN. Further details about SCAN can be found in 
this paper:
> Zaiqiao Meng, Shangsong Liang, Jinyuan Fang, Teng Xiao. Semi-supervisedly Co-embedding Attributed Networks. (NeurIPS 2019)

The orignal tensorflow implementation for SCAN can be found in [SCAN](https://github.com/mengzaiqiao/SCAN)




## Requirements

=================
* Pytorch (1.0 or later)
* python 3.6/3.7
* scikit-learn
* scipy

## Run the demo
=================

```bash
python train.py
```
## Result


The  Link prediction performance AUC&AP score :

| Dataset     |  AUC  |  AP   |
| :---------- | :---: | :---: |
| BLOGCATALOG | 0.844 | 0.850 |
| CORA        | 0.972 | 0.972 |
| FLICKR      | 0.889 | 0.906 |

The  Attribute inference performance AUC&AP score :

| Dataset     |  AUC  |  AP   |
| :---------- | :---: | :---: |
| BLOGCATALOG | 0.886 | 0.888 |
| CORA        | 0.822 | 0.838 |
| FLICKR      | 0.864 | 0.859 |

The  node classification performance accuracy :

| Dataset     |  ACC of SCVA_SVM  |  ACC of SCVA_DIS   |
| :---------- | :---: | :---: |
| BLOGCATALOG | 0.834 | 0.844 |
| CORA        | 0.736 | 0.822 |
| FLICKR      | 0.695 | 0.800 |
