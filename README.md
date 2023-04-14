<p align="center">
    <img src="https://raw.githubusercontent.com/neuml/txtinstruct/master/logo.png"/>
</p>

<h3 align="center">
    <p>Datasets and models for instruction-tuning</p>
</h3>

-------------------------------------------------------------------------------------------------------------------------------------------------------

txtinstruct is a framework for training instruction-tuned models.

![architecture](https://raw.githubusercontent.com/neuml/txtinstruct/master/images/architecture.png#gh-light-mode-only)
![architecture](https://raw.githubusercontent.com/neuml/txtinstruct/master/images/architecture-dark.png#gh-dark-mode-only)

The objective of this project is to support open data, open models and integration with your own data. One of the biggest problems today is the lack of licensing clarity with instruction-following datasets and large language models. txtinstruct makes it easy to build your own instruction-following datasets and use those datasets to train instructed-tuned models.

txtinstruct is built with Python 3.7+ and [txtai](https://github.com/neuml/txtai).

## Installation

The easiest way to install is via pip and PyPI

    pip install txtinstruct

You can also install txtinstruct directly from GitHub. Using a Python Virtual Environment is recommended.

    pip install git+https://github.com/neuml/txtinstruct

Python 3.7+ is supported

See [this link](https://github.com/neuml/txtai#installation) to help resolve environment-specific install issues.

## Examples

The following example notebooks show how to build models with txtinstruct.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Introducing txtinstruct](https://github.com/neuml/txtinstruct/blob/master/examples/01_Introducing_txtinstruct.ipynb) | Build instruction-tuned datasets and models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtinstruct/blob/master/examples/01_Introducing_txtinstruct.ipynb) |
