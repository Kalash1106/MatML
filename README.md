# MatML: A MultiModal LLM for defect classification and defect description[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
ME793 Course Project for Spring 2k24 classifying defective material images and generating a small description of the defective images.

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
