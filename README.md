# DTLP-Release
Tensorflow codes for paper: **Shipeng Zhang, Lizhi Wang, Lei Zhang, and Hua Huang, Learning Tensor Low-Rank Prior for Hyperspectral Image Reconstruction, IEEE CVPR, 2021.**[[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Tensor_Low-Rank_Prior_for_Hyperspectral_Image_Reconstruction_CVPR_2021_paper.pdf)

## Abstract
Snapshot hyperspectral imaging has been developed to capture the spectral information of dynamic scenes. In this paper, we propose a deep neural network by learning the tensor low-rank prior of hyperspectral images (HSI) in the feature domain to promote the reconstruction quality. Our method is inspired by the canonical-polyadic (CP) decomposition theory, where a low-rank tensor can be expressed as a weight summation of several rank-1 component tensors. Specifically, we first learn the tensor low-rank prior of the image features with two steps: (a) we generate rank-1 tensors with discriminative components to collect the contextual information from both spatial and channel dimensions of the image features; (b) we aggregate those rank-1 tensors into a low-rank tensor as a 3D attention map to exploit the global correlation and refine the image features. Then, we integrate the learned tensor low-rank prior into an iterative optimization algorithm to obtain an end-to-end HSI reconstruction. Experiments on both synthetic and real data demonstrate the superiority of our method.

## Data
We provide data of two datasets (*Harvard and ICVL*) for training and testing. It can be download from [Data](https://drive.google.com/drive/). 

The resolution of testing data provided here is $512 \times 512 \times 31$. Note that, for the comparison with previous methods, we evaluate the results of the central areas with $256 \times 256 \times 31$ in the paper.

## Environment
Python 2.7.18<br/>
CUDA 9.0<br/>
Tensorflow 1.11.0<br/>


## Usage
1. Download this repository via git or download the [zip file](https://codeload.github.com/zspCoder/DTLP/main) manually.
```
git clone https://github.com/zspCoder/DTLP
```
2. Download the data from [Data](https://drive.google.com/drive/)

2. Run the file **Train.py** to train the model.

3. Run the file **Test.py** to test the model.

## Citation
@inproceedings{zhang2021learning,<br/>
  title={Learning Tensor Low-Rank Prior for Hyperspectral Image Reconstruction},<br/>
  author={Zhang, Shipeng and Wang, Lizhi and Zhang, Lei and Huang, Hua},<br/>
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},<br/>
  pages={12006--12015},<br/>
  year={2021}<br/>
}