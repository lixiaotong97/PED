# Exploring Model Transferability through the Lens of Potential Energy
Official pytorch implementation of ["Exploring Model Transferability through the Lens of Potential Energy"](https://arxiv.org/abs/2308.15074) in International Conference on Computer Vision (ICCV) 2023.

By [Xiaotong Li](https://scholar.google.com/citations?user=cpCE_T4AAAAJ&hl=zh-CN), [Zixuan Hu](https://github.com/lixiaotong97/PED), [Yixiao Ge](https://geyixiao.com/), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=zh-CN), [Ling-Yu Duan](https://scholar.google.com/citations?user=hsXZOgIAAAAJ&hl=zh-CN).

## Introduction

> Transfer learning has become crucial in computer vision tasks due to the vast availability of pre-trained deep learning models. However, selecting the optimal pre-trained model from a diverse pool for a specific downstream task remains a challenge. Existing methods for measuring the transferability of pre-trained models rely on statistical correlations between encoded static features and task labels, but they overlook the impact of underlying representation dynamics during fine-tuning, leading to unreliable results, especially for self-supervised models. In this paper, we present an insightful physics-inspired approach named PED to address these challenges. We reframe the challenge of model selection through the lens of potential energy and directly model the interaction forces that influence fine-tuning dynamics. By capturing the motion of dynamic representations to decline the potential energy within a force-driven physical model, we can acquire an enhanced and more stable observation for estimating transferability. The experimental results on 10 downstream tasks and 12 self-supervised models demonstrate that our approach can seamlessly integrate into existing ranking techniques and enhance their performances, revealing its effectiveness for the model selection task and its potential for understanding the mechanism in transfer learning.

![Overview](Overview.png)

## Updates
The code is coming soon.

## Reference
```
@inproceedings{li2023exploring,
  title={Exploring Model Transferability through the Lens of Potential Energy},
  author={Li, Xiaotong and Hu, Zixuan and Ge, Yixiao and Shan, Ying and Duan, Ling-Yu},
  booktitle={International Conference on Computer Vision},
  year={2023},
}
```
