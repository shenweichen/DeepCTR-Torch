.. DeepCTR-PyTorch documentation master file, created by
   sphinx-quickstart on Fri Nov 23 21:08:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepCTR-Torch's documentation!
===================================

|Downloads|_ |Stars|_ |Forks|_ |PyPi|_ |Issues|_ |Chat|_

.. |Downloads| image:: https://pepy.tech/badge/deepctr-torch
.. _Downloads: https://pepy.tech/project/deepctr-torch

.. |Stars| image:: https://img.shields.io/github/stars/shenweichen/deepctr-torch.svg
.. _Stars: https://github.com/shenweichen/DeepCTR-Torch

.. |Forks| image:: https://img.shields.io/github/forks/shenweichen/deepctr-torch.svg
.. _Forks: https://github.com/shenweichen/DeepCTR-Torch/fork

.. |PyPi| image:: https://img.shields.io/pypi/v/deepctr-torch.svg
.. _PyPi: https://pypi.org/project/deepctr-torch/

.. |Issues| image:: https://img.shields.io/github/issues/shenweichen/deepctr-torch.svg
.. _Issues: https://github.com/shenweichen/deepctr-torch/issues

.. |Chat| image:: https://img.shields.io/badge/chat-wechat-brightgreen?style=flat
.. _Chat: ./#disscussiongroup

DeepCTR-Torch is a **Easy-to-use** , **Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layer  which can be used to build your own custom model easily.It is compatible with **PyTorch**.You can use any complex model with ``model.fit()`` and ``model.predict()``.

Let's `Get Started! <./Quick-Start.html>`_ (`Chinese Introduction <https://zhuanlan.zhihu.com/p/53231955>`_)

You can read the latest code at https://github.com/shenweichen/DeepCTR-Torch and `DeepCTR <https://github.com/shenweichen/DeepCTR>`_ for tensorflow version.

News
-----
03/27/2020 : Add `DIN <./Features.html#din-deep-interest-network>`_ and `DIEN <./Features.html#dien-deep-interest-evolution-network>`_ . `Changelog <https://github.com/shenweichen/DeepCTR-Torch/releases/tag/v0.2.1>`_

01/31/2020 : Refactor `feature columns <./Features.html#feature-columns>`_ . Support double precision in metric calculation . `Changelog <https://github.com/shenweichen/DeepCTR-Torch/releases/tag/v0.2.0>`_

10/03/2019 : Simplify the input logic(`examples <./Examples.html#classification-criteo>`_). `Changelog <https://github.com/shenweichen/DeepCTR-Torch/releases/tag/v0.1.3>`_


DisscussionGroup
-----------------------

公众号：**浅梦的学习笔记**  wechat ID: **deepctrbot**

.. image:: ../pics/weichennote.png

.. toctree::
   :maxdepth: 2
   :caption: Home:

   Quick-Start<Quick-Start.md>
   Features<Features.md>
   Examples<Examples.md>
   FAQ<FAQ.md>
   History<History.md>

.. toctree::
   :maxdepth: 3
   :caption: API:

   Models<Models>
   Layers<Layers>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
