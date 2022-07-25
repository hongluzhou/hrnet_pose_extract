# Keypoint Extraction with HRNet on the Volleyball / Collective Activity Dataset
The project is an official implementation of the keypoint extraction part of our ECCV 2022 paper [COMPOSER: Compositional Reasoning of Group Activity in Videos with Keypoint-Only Modality](https://arxiv.org/abs/2112.05892).  
  
If you find our repo useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@article{zhou2022composer,
  title={COMPOSER: Compositional Reasoning of Group Activity in Videos with Keypoint-Only Modality},
  author={Zhou, Honglu and Kadav, Asim and Shamsian, Aviv and Geng, Shijie and Lai, Farley and Zhao, Long and Liu, Ting and Kapadia, Mubbasir and Graf, Hans Peter},
  journal={Proceedings of the 17th European Conference on Computer Vision (ECCV 2022)},
  year={2022}
}
```  

# Installation
Please follow the instructions [here](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch#installation) for installation.

# Preparation after Installation
- Step 1: please clone the official HRNet [repo](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
```shell
git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
```
- Step 2: download the pretrained HRNet model checkpoints using [this link](https://drive.google.com/file/d/1bapNbKc3fd1aG4mugK_pTWHbFx3oMZqK/view?usp=sharing), and put the downloaded folder `models` inside the `deep-high-resolution-net.pytorch/` directory.      
- Step 3: replace the `demo` folder in the original `deep-high-resolution-net.pytorch` repository with the `demo` folder from this repository.      
We have also put the code together and uploaded to [this link](https://drive.google.com/file/d/1L2MQsDsV-i7y75t8Yv7lcqmR7OXUcIyC/view?usp=sharing) which you can refer to if you have any questions on the preparation.  
  
# Keypoint Extraction on Volleyball
```shell
deep-high-resolution-net.pytorch$ cd demo
deep-high-resolution-net.pytorch/demo$ python volleyball_joint_feature_extraction.py --dataset_path path_to_volleyball_videos --track_path path_to_tracks_normalized --save_path path_to_save_extracted_keypoints
```
# Keypoint Extraction on Collective Activity
```shell
deep-high-resolution-net.pytorch$ cd demo
deep-high-resolution-net.pytorch/demo$ python collective_joint_feature_extraction.py --dataset_path path_to_collective_activity_videos --track_path path_to_tracks_normalized --save_path path_to_save_extracted_keypoints
```
  

