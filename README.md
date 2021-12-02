# Reproducibility Study for the Paper: "KiU-Net: Overcomplete Convolutional Architectures forBiomedical Image and Volumetric Segmentation"

This repository is for the reproducibility study of [KiU-Net: Overcomplete Convolutional Architectures forBiomedical Image and Volumetric Segmentation](https://arxiv.org/pdf/2010.01663v2.pdf). 

This repository reflects our most up-to-date work in reproducing the paper.

## Prerequisites:
- Python 3.7 and above*
- Tensorflow 2.X.X

\*We recommend using Anaconda (Python 3.8)

<a href="https://www.python.org/downloads/"> Python Installation </a>  
<a href="https://www.anaconda.com/products/individual"> Anaconda Installation </a>  
<a href="https://www.tensorflow.org/install"> Tensorflow Installation </a>  

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Links for downloading the public Datasets:

RITE Dataset - <a href = "https://drive.google.com/drive/folders/1WTPRJk8Q-Bx-uqMyfoL9JHi7vKotwgL8?usp=sharing"> Link (Resized) </a>

## Training

To train the model(s) in the paper, run this command:

```train
python train_model.py
```

The models will be saved to a in a local subdirectory when training is completed.


Alternatively, you may run the cells in the Jupyter notebook on your local machine.

## Evaluation

To evaluate the model on the RITE dataset, run:

```eval
python evaluation.py
```

Alternatively, you may run the cells in the Jupyter notebook on your local machine.

## Results

Our model achieves the following performance on :

### [Image Classification on RITE dataset](https://paperswithcode.com/sota/medical-image-segmentation-on-rite)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| KiU-Net   |     TBD         |      TBD       |
