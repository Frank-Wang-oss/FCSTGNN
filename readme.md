# Pytorch implementation of [Fully-Connected Spatial-Temporal Graph Neural Network for Multivariate Time-Series Data](https://arxiv.org/pdf/2309.05305.pdf). 

By: Yucheng Wang, [Yuecong Xu](https://xuyu0010.github.io/), [Jianfei Yang](https://marsyang.site/), [Min Wu](https://sites.google.com/site/wumincf/), [Xiaoli Li](https://personal.ntu.edu.sg/xlli/), [Lihua Xie](https://personal.ntu.edu.sg/elhxie/), [Zhenghua Chen](https://zhenghuantu.github.io/)

# :boom: Our paper has been accepted for publication of AAAI 2024 (acceptance rate 23.75%).

# Requirements

You will need the following to run the above:
- Pytorch 1.13.0, Torchvision 0.14.0
- Python 3.6.8, Pillow 5.4.1, Numpy 1.22.4
- If you want to train (and don't want to wait for 4 months):
  - A decent GPU
  - All the required NVIDIA software to run PyTorch on a GPU (cuda, etc)

# Abstract

Multivariate Time-Series (MTS) data is crucial in various application fields. With its sequential and multi-source (multiple sensors) properties, MTS data inherently exhibits Spatial-Temporal (ST) dependencies, involving temporal correlations between timestamps and spatial correlations between sensors in each timestamp. To effectively leverage this information, Graph Neural Network-based methods (GNNs) have been widely adopted. However, existing approaches separately capture spatial dependency and temporal dependency and fail to capture the correlations between Different sEnsors at Different Timestamps (DEDT). Overlooking such correlations hinders the comprehensive modelling of ST dependencies within MTS data, thus restricting existing GNNs from learning effective representations. To address this limitation, we propose a novel method called Fully-Connected Spatial-Temporal Graph Neural Network (FC-STGNN), including two key components namely FC graph construction and FC graph convolution. For graph construction, we design a decay graph to connect sensors across all timestamps based on their temporal distances, enabling us to fully model the ST dependencies by considering the correlations between DEDT. Further, we devise FC graph convolution with a moving-pooling GNN layer to effectively capture the ST dependencies for learning effective representations. Extensive experiments show the effectiveness of FC-STGNN on multiple MTS datasets compared to SOTA methods. The code is available at https://github.com/Frank-Wang-oss/FCSTGNN.

![1702895493630](https://github.com/Frank-Wang-oss/FCSTGNN/assets/73806631/7e03095b-b954-43d4-94e0-3b0434db5e2f)

# Dataset

We use three datasets to evaluate our method, including C-MAPSS, UCI-HAR, and ISRUC-S3.

## C-MAPSS

You can access [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/), and put the downloaded dataset into directory 'CMAPSSData'.

For running the experiments on C-MAPSS, directly run main_RUL.py

## UCI-HAR

You can access [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones), and put the downloaded dataset into directory 'HAR'.

For running the experiments on UCI-HAR, you need to first run preprocess_UCI_HAR.py to pre-process the dataset. After that, run main_HAR.py

## ISRUC-S3
 
You can access [here](https://sleeptight.isr.uc.pt/), and download S3 and put the downloaded dataset into directory 'ISRUC'.

For running the experiments on ISRUC, you need to first run preprocess_ISRUC.py to pre-process the dataset. After that, run main_ISRUC.py

## Experimental Results

![1702895532299](https://github.com/Frank-Wang-oss/FCSTGNN/assets/73806631/737ab9da-2eab-4f32-8126-b3f237d0bc53)

# Citation

```
@article{Wang2023FullyConnectedSG,
  title={Fully-Connected Spatial-Temporal Graph for Multivariate Time Series Data},
  author={Yucheng Wang and Yuecong Xu and Jianfei Yang and Min Wu and Xiaoli Li and Lihua Xie and Zhenghua Chen},
  journal={ArXiv},
  year={2023},
  volume={abs/2309.05305},
  url={https://api.semanticscholar.org/CorpusID:261682449}
}
```

# Acknowledgement

We thank the codes of preprocessing for [UCI-HAR](https://github.com/emadeldeen24/TS-TCC) and [ISRUC-S3](https://github.com/ziyujia/MSTGCN).
