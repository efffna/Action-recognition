# Action Recognition


## Video Swin Transformer

Recognizer3D : backbone - SwinTransformer3D, cls_head - I3D

За основу взята реализация, основанная на mmaction2, где также рассмотрены c3d, csn, i3d, slowfast, swin, tanet, tin, tpn, trn, tsm, tsn, omnisource, r2plus1d подходы

![teaser](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/figures/teaser.png)
Базовая структура Video Swin Transformer очень близка к структуре SwinTransformer с добавлением измерения кадра, времени(T) при расчете модели

Step:

1. Video to token
2. Model stages 
3. Head - classification (I3DHead)

### Датасет
Kinetics 700-2020

class_name = [
    "belly_dancing",
    "breakdancing",
    "country_line_dancing",
    "dancing_ballet",
    "dancing_charleston",
    "dancing_gangnam_style",
    "dancing_macarena",
    "jumpstyle_dancing",
    "mosh_pit_dancing",
    "robot_dancing",
    "salsa_dancing",
    "square_dancing",
    "swing_dancing",
    "tango_dancing",
    "tap_dancing",
]

```
python3 preparation_data.py
```

### Запуск
```
export PYTHONPATH="${PYTHONPATH}":pwd
```
```
python3 tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py
```
```
python3 tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE>
```
### Метрики

```
Evaluating top_k_accuracy ...
top1_acc   0.6460
top5_acc   0.9033

Evaluating mean_class_accuracy ...
mean_class_accuracy: 0.6418
```



## TSN Resnet
Recognizer2D :  backbone - ResNet50, cls_head - TSN

![teaser](https://user-images.githubusercontent.com/34324155/143019237-8823045b-dfa3-45cc-a992-ee83ab9d8459.png)

Видео делится на сегменты кадров, из каждого сегмента случайным образом выбирается короткий фрагмент

### Метрики

```
Evaluating top_k_accuracy ...
top1_acc: 0.5203
top5_acc: 0.8375

Evaluating mean_class_accuracy ...
mean_class_accuracy: 0.5230
```
