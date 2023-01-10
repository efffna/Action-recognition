# Action Recognition


## 2d net

Для классификации видеороликов с использованием обычной сверточной сети (У нас resnet18) применим следующий алгоритм:

* Нарежем обучающие видеоролики на кадры с шагом n (У нас с шагом 20)
* Обучим модель предсказывать кадр из видеоролика 
* Тестируем сеть:
  * Из тестового ролика предсказываем каждый m кадр (у нас 5)
  * Суммируем все выходы модели из каждого кадра
  * Присваиваем метку с наибольшим значением

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

## Запуск
```
Training:
    python3 train_2d_net.py
Testing:
    python3 test_2d_net.py
Metrics:
    jupyter notebook preparation_and_metrics.ipynb
```

## Метрики
```
                   precision   recall    f1-score
    accuracy                             0.34
   macro avg       0.32        0.31      0.32
weighted avg       0.35        0.34      0.34
```
