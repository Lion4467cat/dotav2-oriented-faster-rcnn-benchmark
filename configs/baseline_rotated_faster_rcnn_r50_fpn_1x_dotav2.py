"""Track A: Vanilla Rotated Faster R-CNN (ResNet-50 + FPN, 1x) baseline on DOTA v2.0.

Inherits MMRotate dev-1.x's official rotated-faster-rcnn baseline and adapts
dataset classes (18 DOTA v2.0 categories), paths, AMP, and checkpointing for
Google Colab Pro/Pro+.
"""

_base_ = [
    "../mmrotate/configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py"
]

angle_version = "le90"

DOTA_V2_CLASSES = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
    "container-crane",
    "airport",
    "helipad",
)

DOTA_V2_PALETTE = [
    (165, 42, 42),
    (189, 183, 107),
    (0, 255, 0),
    (255, 0, 0),
    (138, 43, 226),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 193, 193),
    (0, 51, 153),
    (255, 250, 205),
    (0, 139, 139),
    (255, 255, 0),
    (147, 116, 116),
    (0, 0, 255),
    (100, 149, 237),
    (34, 139, 34),
    (255, 20, 147),
]

metainfo = dict(classes=DOTA_V2_CLASSES, palette=DOTA_V2_PALETTE)

# Default to Colab local NVMe patched directory; override via --cfg-options if needed.
data_root = "/content/data/split_ss_dotav2/"

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(DOTA_V2_CLASSES)),
    )
)

train_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=None),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="mmdet.RandomFlip",
        prob=0.75,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(type="mmdet.PackDetInputs"),
]

val_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=None),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="DOTADataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train/annfiles/",
        data_prefix=dict(img_path="train/images/"),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DOTADataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val/annfiles/",
        data_prefix=dict(img_path="val/images/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="DOTAMetric", metric="mAP")
test_evaluator = val_evaluator

# Enable AMP for Colab A100/L4 speedup with gradient clipping for stability.
optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=2,  # Effective batch size = 4 * 2 = 8
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=2)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="dota/mAP",
        rule="greater",
    )
)
