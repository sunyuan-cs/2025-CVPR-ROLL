# 2025-CVPR-ROLL

PyTorch implementation for ''ROLL: Robust Noisy Pseudo-label Learning for Multi-View Clustering with Noisy Correspondence'' (CVPR 2025).

## Requirements

pytorch==1.5.0 

numpy>=1.18.2

scikit-learn>=0.22.2

munkres>=1.1.2

logging>=0.5.1.2

## Datasets

The used datasets could be downloaded from quark (链接：https://pan.quark.cn/s/fd293cb3bea7 提取码：y1Pz).

## Demo

Train a model with different settings

<!-- 多视图聚类，关联噪声 -->
python test_roll.py --data 4 



## Citation

If you find our work useful in your research, please consider citing:

```latex

@inproceedings{sun2025roll,
  title={ROLL: Robust Noisy Pseudo-label Learning for Multi-View Clustering with Noisy Correspondence},
  author={Sun, Yuan and Li, Yongxiang and Ren, Zhenwen and Duan, Guiduo and Peng, Dezhong and Hu, Peng},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30732--30741},
  year={2025}
}


@ARTICLE{sun2024RMCNC,
  author={Sun, Yuan and Qin, Yang and Li, Yongxiang and Peng, Dezhong and Peng, Xi and Hu, Peng},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Robust Multi-View Clustering With Noisy Correspondence}, 
  year={2024},
  volume={36},
  number={12},
  pages={9150-9162}}


@ARTICLE{yang2022SURE,
  author={Yang, Mouxing and Li, Yunfan and Hu, Peng and Bai, Jinfeng and Lv, Jiancheng and Peng, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Robust Multi-View Clustering With Incomplete Information}, 
  year={2023},
  volume={45},
  number={1},
  pages={1055-1069}}
```
