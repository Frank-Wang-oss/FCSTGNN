Pytorch implementation of [Fully-Connected Spatial-Temporal Graph Neural Network for Multivariate Time-Series Data]. 

# Requirements

You will need the following to run the above:
- Pytorch 1.9.1, Torchvision 0.10.1
- Python 3.6.8, Pillow 5.4.1
- If you want to train (and don't want to wait for 4 months):
  - A decent GPU
  - All the required NVIDIA software to run PyTorch on a GPU (cuda, etc)
  
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

# Acknowledgement

We thank the codes of preprocessing for [UCI-HAR](https://github.com/emadeldeen24/TS-TCC) and [ISRUC-S3](https://github.com/ziyujia/MSTGCN).