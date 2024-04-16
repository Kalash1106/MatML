# MatML: A MultiModal LLM for defect classification and defect description[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
<b>ME793 (Multiscale Materials Informatics, Discovery and Design) </b> Course Project under <a href="https://www.me.iitb.ac.in/?q=faculty/Prof.%20Alankar%20Alankar">Prof. Alankar</a> for Spring 2k24. The objective of the project is to classify defective material images and generate a technical description of the defective images using existing Large Language Models.

## Repo Structure & Walkthrough
- The folder [EDA_Embeddings](https://github.com/Kalash1106/MatML/tree/main/EDA_Embeddings) consists of all the files for Exploratory Data Analysis. It also contains all the codes for embedding analysis and contrastive learning.
- The folder [defect_detection](https://github.com/Kalash1106/MatML/tree/main/defect_detection) consists of all the codes for training the ResNet34 model for (binary) defect detection.
- The folder [multi_modal_llm](https://github.com/Kalash1106/MatML/tree/main/multi_modal_llm) consists of all the codes for finetuning the BLIP2 model to adapt it for defect description generation.
- The folder [Testing images](https://github.com/Kalash1106/MatML/tree/main/Testing%20images) consists of all the images for evaluating the model
- The file [main_demo.ipynb](https://github.com/Kalash1106/MatML/tree/main/main_demo.ipynb) is the code for demonstration hosted via gradio.


## Setup
First clone the repository and install the necessary libraries using the requirements file.
```sh
python3 -m venv .venv
git clone --recursive https://github.com/Kalash1106/MatML.git
pip install -r requirements.txt
```

## Running the Demo file
1. Download the weights of the trained ResNet34 model from https://drive.google.com/file/d/1cmWYZJUlDnoH4rScLoO4rJKw-JIhVgGb/view?usp=drive_link.
2. Generate your Google AI API key following the instructions given at https://aistudio.google.com/app/apikey.
3. Enter the correct path and the api key at the relevant locations in the <i> main_demo.ipynb </i> notebook and execute all the cells.


## Dataset
The dataset.zip file (consisting of the entire dataset) can be downloaded at https://tinyurl.com/42nzb2pz (~ 1.1 GB of disk space) . <br>This dataset has been adopted from two sources, Crack Defects (crack) and Kolektor Surface Defect Detection 2 (ksdd), detailed below.
### Directory Layout
    .
    ├── crack_defective
    ├── crack_non_defective
    ├── ksdd_defective       
    └── ksdd_non_defective 
    
### Surface Crack Detection


- The dataset contains concrete images having cracks. The data is collected from various METU Campus Buildings.
- The dataset is divided into two as negative and positive crack images for image classification. 
- Each class has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels. 
- The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016). 
High-resolution images have variance in terms of surface finish and illumination conditions. 
- <i>Citation</i> - Özgenel, Çağlar Fırat (2019), “Concrete Crack Images for Classification”, Mendeley Data, V2, doi: 10.17632/5y9wdsg2zt.2
- <i>Source Link</i> - https://data.mendeley.com/datasets/5y9wdsg2zt/2 


### Kolektor SDD2
The dataset consists of
- 356 images with visible defects and 2979 images without any defect
- image sizes of approximately 230 x 630 pixels
- several different types of defects (scratches, minor spots, surface imperfections, etc.)
- <i>Citation</i> - Božič, Jakob, Domen Tabernik, and Danijel Skočaj. "Mixed supervision for surface-defect detection: From weakly to fully supervised learning." Computers in Industry 129 (2021): 103459.
- <i>Source Link </i> - https://www.vicos.si/resources/kolektorsdd2/
## Contributors
- Kalash Shah (200100079)
- Ved Khandekar (20d170019)
- Yash Kothari (200100174)
