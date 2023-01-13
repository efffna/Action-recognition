_base_ = [
    "../../models/tsn_r50/tsn_r50.py",
    "../../models/schedules/sgd_100e.py",
    "../../models/default_runtime.py",
]


# dataset settings
dataset_type = "VideoDataset"
data_root = "data/train"
data_root_val = "data/val"
ann_file_train = "data/train.txt"
ann_file_val = "data/val.txt"
ann_file_test = "data/test.txt"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
    ),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]
val_pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=2, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="ThreeCrop", crop_size=256),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, ann_file=ann_file_train, data_prefix=data_root, pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type, ann_file=ann_file_val, data_prefix=data_root_val, pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=5, metrics=["top_k_accuracy", "mean_class_accuracy"])
total_epochs = 20
optimizer = dict(type="SGD", lr=0.000375, momentum=0.9, weight_decay=0.0001)

lr_config = dict(policy="step", step=[8])

# runtime settings
work_dir = "./work_dirs/tsn_r50/"
