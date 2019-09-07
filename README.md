# DeepCTR-Pytorch

[![Python Versions](https://img.shields.io/pypi/pyversions/deepctr.svg)](https://pypi.org/project/deepctr)
[![Downloads](https://pepy.tech/badge/deepctr)](https://pepy.tech/project/deepctr)
[![GitHub Issues](https://img.shields.io/github/issues/shenweichen/deepctr-pytorch.svg
)](https://github.com/shenweichen/deepctr-pytorch/issues)


[![Documentation Status](https://readthedocs.org/projects/deepctr-doc/badge/?version=latest)](https://deepctr-doc.readthedocs.io/)
[![Gitter](https://badges.gitter.im/DeepCTR/community.svg)](https://gitter.im/DeepCTR/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![License](https://img.shields.io/github/license/shenweichen/deepctr.svg)](https://github.com/shenweichen/deepctr/blob/master/LICENSE)

PyTorch version of [DeepCTR](https://github.com/shenweichen/DeepCTR).

DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layers which can be used to build your own custom model easily.You can use any complex model with `model.fit()`and `model.predict()` .

Let's [**Get Started!**](https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html)([Chinese Introduction](https://zhuanlan.zhihu.com/p/53231955))

## **Call for contributers**

This project is under development and we need developers to participate in the construction.

If you 
- familiar and interested in CTR models
- familiar with pytorch(both pytorch and tensorflow better)
- have spare time to learn and develop
- familiar with git

please send a brief introduction of your background and experience to wcshen1994@163.com, welcome to join us!

## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Convolutional Click Prediction Modelin progress】 | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)               |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|                AutoInt                 | [arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|                  NFFM       | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                           |
|                  FGCNN                  | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                            |
|                  Deep Session Interest Network                | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                         |
|                  FiBiNET               | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |