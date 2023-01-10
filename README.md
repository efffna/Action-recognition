# Action Recognition


## Video Swin Transformer
За основу взята официальная реализация ["Video Swin Transformer"](https://github.com/SwinTransformer/Video-Swin-Transformer).
где также рассмотрены c3d, csn, i3d, slowfast, swin, tanet, tin, tpn, trn, tsm, tsn, omnisource, r2plus1d

![teaser](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/figures/teaser.png)
Базовая структура Video Swin Transformer очень близка к структуре SwinTransformer с добавлением измерения кадра, времени(T) при расчете модели
Step:

1. Video to token
2. Model stages 
3. Head - classification (I3DHead)

## Датасет
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


## Запуск
```
python3 tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py --cfg-options load_from=swin_tiny_patch244_window877_kinetics400_1k.pth
```

## Метрики
```
Evaluating top_k_accuracy ...
top1_acc   0.6460
top5_acc   0.9033

Evaluating mean_class_accuracy ...
mean_acc   0.6418
mean_class_accuracy: 0.6418
```