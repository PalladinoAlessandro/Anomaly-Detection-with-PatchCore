# Anomaly-Detection-with-PatchCore

This project aims to reproduce and compare the results of the anomaly detection with PatchCore by using both ImageNet and CLIP.

In the following sections the instruction to run it:
\* as the analysis was made on Google Colab, the instruction will refer to it 

---

## Usage

Mount:
```shell
from google.colab import drive
drive.mount('/content/drive')
```
Go to the desired folder:
```shell
cd /content/drive/MyDrive/'desired_folder'/
```
Cloning from GitHub:
```shell
!git clone https://github.com/PalladinoAlessandro/Anomaly-Detection-with-PatchCore/tree/main.git
```
Go to the folder "code":
```shell
cd /content/drive/MyDrive/'desired_folder'/code
```
Install the requirements:
```shell
$ !pip install --upgrade pip
$ !pip install -r requirements.txt
$ !pip install --upgrade git+https://github.com/openai/CLIP.git
```

CLI:
```shell
$ !python indad/run.py
# choose among the given options the desired Class and Backbone
```
Results can be found under `./results/`.

<details>
  <summary> 👁️ </summary>

### Datasets

The datasets folder and the needed subfolders are created by running the code. Note that you need at least 5 gb of free space to download all the datasets, plus additional space since the code will generate output images highlighting, where present, the anomalous regions in the test samples.

Here the link to download the MVTec datasets if you prefer to get them manually:

https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz
https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz
https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz

\* for our specific analysis we used a reduced version of the following classes, due to computational power limitations related to the GPU provided by Google Colab:
  1. carpet
  2. grid
  3. hazelnut
  4. screw

You will find these three classes in the "datasets" folder.

Check out one of the downloaded MVTec datasets (or the three "_reduced").
Naming of images should correspond among folders.
Right now there is no support for no ground truth pixel masks.

```
📂 datasets
 ┗📂 dataset_name
  ┣ 📂 ground_truth/defective
  ┃ ┣ 📂 defect_type_1
  ┃ ┗ 📂 defect_type_2
  ┣ 📂 test
  ┃ ┣ 📂 defect_type_1
  ┃ ┣ 📂 defect_type_2
  ┃ ┗ 📂 good
  ┗ 📂 train/good
```

After running the main script, it will create a folder for each backbone utilized:
```
📂 datasets
 ┗📂 dataset_name
  ┣ 📂 ground_truth/defective
  ┃ ┣ 📂 defect_type_1
  ┃ ┗ 📂 defect_type_2
  ┣ 📂 test
  ┃ ┣ 📂 defect_type_1
  ┃ ┣ 📂 defect_type_2
  ┃ ┗ 📂 good
  ┣ 📂 train/good
  ┗ 📂 output_backbone_name
```

---

## Results

📝 = paper, 👇 = reference repo

### Image-level average % Score
  
```  
  📝 =  99.1
  👇 = 97.7
  WideResNet50 =  97.9
  ResNet50 = 97.8
  ResNet101 = 96.5
```

### Pixel-level average % Score

```
  📝 =  98.1
  👇 = 97.2
  WideResNet50 =  97.8
  ResNet50 = 97.5
  ResNet101 = 96.9
```

### Hyperparams

The following parameters were used to calculate the results. 
They more or less correspond to the parameters used in the papers.
\* exepction made for the backbone that varies according to the net used for the training phase

```yaml
patchcore:
  backbones: wide_resnet50_2 , ResNet50 , ResNet101
  f_coreset: 0.1
  n_reweight: 3
```

---

## References

PatchCore:
```bibtex
@InProceedings{Roth_2022_CVPR,
    author    = {Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch\"olkopf, Bernhard and Brox, Thomas and Gehler, Peter},
    title     = {Towards Total Recall in Industrial Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14318-14328}
}

@misc{rvorias22,
  author = {rvorias dham and h1day},
  title = {Industrial KNN-based Anomaly Detection},
  year = {2022},
  note = {\url{https://github.com/rvorias/ind_knn_ad}}
}

@misc{Carluccio23,
  author = {Alex Carluccio and Luigi Federico and Samuele Longo},
  title = {PatchCore for Industrial Anomaly Detection},
  year = {2023},
  note = {\url{https://github.com/LuigiFederico/PatchCore-for-Industrial-Anomaly-Detection}}
}

@article{DBLP:journals/corr/abs-2103-00020,
  author = {Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and  Gretchen Krueger and Ilya Sutskever},
  title = {Learning Transferable Visual Models From Natural Language Supervision},
  journal = {CoRR},
  volume = {abs/2103.00020},
  year = {2021},
  url = {https://arxiv.org/abs/2103.00020},
  eprinttype = {arXiv},
  eprint = {2103.00020},
  timestamp = {Thu, 04 Mar 2021 17:00:40 +0100},
  biburl = {https://dblp.org/rec/journals/corr/abs-2103-00020.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/abs-1803-07728,
  author       = {Spyros Gidaris and
                  Praveer Singh and
                  Nikos Komodakis},
  title        = {Unsupervised Representation Learning by Predicting Image Rotations},
  journal      = {CoRR},
  volume       = {abs/1803.07728},
  year         = {2018},
  url          = {http://arxiv.org/abs/1803.07728},
  eprinttype    = {arXiv},
  eprint       = {1803.07728},
  timestamp    = {Mon, 13 Aug 2018 16:46:04 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1803-07728.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/abs-2002-05709,
  author       = {Ting Chen and
                  Simon Kornblith and
                  Mohammad Norouzi and
                  Geoffrey E. Hinton},
  title        = {A Simple Framework for Contrastive Learning of Visual Representations},
  journal      = {CoRR},
  volume       = {abs/2002.05709},
  year         = {2020},
  url          = {https://arxiv.org/abs/2002.05709},
  eprinttype    = {arXiv},
  eprint       = {2002.05709},
  timestamp    = {Fri, 14 Feb 2020 12:07:41 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2002-05709.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/abs-1905-04899,
  author       = {Sangdoo Yun and
                  Dongyoon Han and
                  Seong Joon Oh and
                  Sanghyuk Chun and
                  Junsuk Choe and
                  Youngjoon Yoo},
  title        = {CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
                  Features},
  journal      = {CoRR},
  volume       = {abs/1905.04899},
  year         = {2019},
  url          = {http://arxiv.org/abs/1905.04899},
  eprinttype    = {arXiv},
  eprint       = {1905.04899},
  timestamp    = {Tue, 28 May 2019 12:48:08 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1905-04899.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{8954181,
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection}, 
  year={2019},
  volume={},
  number={},
  pages={9584-9592},
  doi={10.1109/CVPR.2019.00982}
}

@article{WANG2022100609,
title = {Evaluating computing performance of deep neural network models with different backbones on IoT-based edge and cloud platforms},
journal = {Internet of Things},
volume = {20},
pages = {100609},
year = {2022},
issn = {2542-6605},
doi = {https://doi.org/10.1016/j.iot.2022.100609},
url = {https://www.sciencedirect.com/science/article/pii/S2542660522000919},
author = {Xiaoxuan Wang and Feiyu Zhao and Ping Lin and Yongming Chen}
}

@misc{elharrouss2022backbonesreview,
      title={Backbones-Review: Feature Extraction Networks for Deep Learning and Deep Reinforcement Learning Approaches}, 
      author={Omar Elharrouss and Younes Akbari and Noor Almaadeed and Somaya Al-Maadeed},
      year={2022},
      eprint={2206.08016},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
