# CTVIS: Consistent Training for Online Video Instance Segmentation

<!-- https://user-images.githubusercontent.com/42159793/234478815-3ca313da-828c-4a6e-ae81-727bad8d84cf.mp4 -->
https://github.com/KainingYing/CTVIS/assets/42159793/25273241-6642-46b4-9a47-bd19beec6f68.mp4
## News

- [2023/07/14] Our work CTVIS is accepted by [**ICCV 2023**](https://iccv2023.thecvf.com/)! Congrats! ✌️
- [2023/07/24] We will release the code ASAP. Stay tuned !

## MODEL ZOO

### YouTube-VIS 2019

| Model | Backbone             | AP   | AP $_{50}$ | AP $_{75}$ | AR $_{1}$ | AR $_{10}$ | Link                                                                                                                          |
|-------| -------------------- | ---- |-----------| ---- | ---- | ---- |-------------------------------------------------------------------------------------------------------------------------------|
| CTVIS | ResNet-50            | 55.1 | 78.2      | 59.1 | 51.9 | 63.2 | config / [weight](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_r50_ytvis19_55.1.pth) / [submission](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_r50_ytvis19_55.1.zip) |
| CTVIS | Swin-L (200 queries) | 65.6 | 87.7      | 72.2 | 56.5 | 70.4 | TODO                                                                                                                          |

### YouTube-VIS 2021

| Model | Backbone             | AP   | AP $_{50}$ | AP $_{75}$ | AR $_{1}$ | AR $_{10}$ | Link                                                                                                                                                                                                                   |
| ----- | -------------------- | ---- | --------- | --------- | -------- | --------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CTVIS | ResNet-50            | 50.1 | 73.7      | 54.7      | 41.8     | 59.5      | config / [weight](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_r50_ytvis21_50.1.pth) / [submission](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_r50_ytvis21_50.1.zip) |
| CTVIS | Swin-L (200 queries) | 61.2 | 84        | 68.8      | 48       | 65.8      | config / [weight](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_swinl_ytvis21_61.2.pth) / [submission](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_swinl_ytvis21_61.2.zip)                                                                                                                                                                                   |

### YouTube-VIS 2022

**Note**: YouTube-VIS 2022 shares the same training set as YouTube-VIS 2021.

| Model | Backbone             | AP   | AP $^{S}$ | AP $^{L}$ | Link                                    |
| ----- | -------------------- | ---- | ---- | ---- | --------------------------------------- |
| CTVIS | ResNet-50         | 44.9 | 50.3 | 39.4 | config / [weight](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_r50_ytvis21_50.1.pth)        |
| CTVIS | Swin-L (200 queries) | 53.8 | 61.2 | 46.4 | config / [weight](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_swinl_ytvis21_61.2.pth)       |

### OVIS

| Model | Backbone             | AP   | AP $_{50}$ | AP $_{75}$ | AR $_{1}$ | AR$_{10}$ | Link                                    |
| ----- | -------------------- | ---- | --------- | --------- | -------- | --------- | --------------------------------------- |
| CTVIS | ResNet-50            | 35.5 | 60.8      | 34.9      | 16.1     | 41.9      | config / [weight]() / [submission](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_r50_ovis_35.5.zip)         |
| CTVIS | Swin-L (200 queries) | 46.9 | 71.5      | 47.5      | 19.1     | 52.1      | config / [weight](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_swinl_ovis_46.9.pth) / [submission](https://github.com/KainingYing/storage/releases/download/CTVIS/ctvis_swinl_ovis_46.9.zip)          |
