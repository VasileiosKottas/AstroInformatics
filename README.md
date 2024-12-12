# AstroInformatics
## Purpose of the Project
The purpose of this project is to use various Deep Neural Networks to predict galaxy formation in high redshifts

## Data Generation
This project generates the photometry and spectrometry of a Galaxy using AGN models, the Starburst and the Spheroid. 
These models create the spectrometry and using different filters in various redshifts, the photometry is created.

# Generative Adversarial Networks (GANs) 
This model uses the method of [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html).
The structure of the model is:

<img src="Pictures/image-2.png" width="440">

# Diffusion Model
This models uses the method of [Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting](https://paperswithcode.com/paper/predict-refine-synthesize-self-guiding)

![alt text](Pictures/image-1.png)

# How to use
## Generate data
To generate the dataset first run the requirements_data.txt and then in the main.py select the number of galaxies to generate. The filters to generated the photometry and the AGN type.

```pip install requirements_data.txt```

```python -m main.py```

The generated data will be in the folder data.

@software{2024ascl.soft06003V,
   author = {{Varnava}, Charalambia and {Efstathiou}, Andreas},
    title = "{SMART: Spectral energy distribution (SED) fitter}",
  howpublished = {Astrophysics Source Code Library, record ascl:2406.003},
     year = 2024,
   eid = {ascl:2406.003},
}