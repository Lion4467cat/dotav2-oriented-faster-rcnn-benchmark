"""Track B: Context-Aware Oriented Faster R-CNN on DOTA v2.0.

Architecture Stack:
- Backbone: ConvNeXt-Small (ImageNet-1k/22k pretrained, 7x7 depthwise kernels)
- Neck: PAFPN (top-down + bottom-up path aggregation for tiny & large aerial objects)
- RPN: OrientedRPNHead (midpoint-offset oriented proposals)
- RoI Head: OrientedStandardRoIHead + RoIAlignRotated + RotatedShared2FCBBoxHead
- Box Loss: GDLoss (KLD Gaussian distribution loss, reg_decoded_bbox=True)
- Imbalance Mitigation: ClassBalancedDataset (oversample_thr=1e-3)
- Augmentations: H/V/D RandomFlip + RandomRotate
- Schedule: 2x (24 epochs), AdamW + CosineAnnealingLR + AMP fp16
"""

_base_ = [
    "../mmrotate/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py"
]

custom_imports = dict(imports=["mmpretrain.models"], allow_failed_imports=False)

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
data_root = "/content/data/split_ss_dotav2/"

convnext_s_ckpt = (
    "https://download.openmmlab.com/mmclassification/v0/convnext/"
    "convnext-small_3rdparty_32xb128_in1k_20220124-d39b5192.pth"
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="mmpretrain.ConvNeXt",
        arch="small",
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=convnext_s_ckpt,
            prefix="backbone.",
        ),
    ),
    neck=dict(
        _delete_=True,
        type="mmdet.PAFPN",
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(DOTA_V2_CLASSES),
            reg_decoded_bbox=True,
            loss_bbox=dict(
                _delete_=True,
                type="GDLoss",
                loss_type="kld",
                fun="log1p",
                tau=1.0,
                sqrt=False,
                loss_weight=5.0,
            ),
        )
    ),
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
    dict(
        type="RandomRotate",
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11],  # storage-tank, roundabout
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

raw_train_dataset = dict(
    type="DOTADataset",
    data_root=data_root,
    metainfo=metainfo,
    ann_file="train/annfiles/",
    data_prefix=dict(img_path="train/images/"),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        _delete_=True,
        type="mmdet.ClassBalancedDataset",
        oversample_thr=1e-3,
        dataset=raw_train_dataset,
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

# OptimWrapper: on A100-80GB, full fp32 precision is used to ensure GaussianDistLoss
# (KLD matrix inversion/determinant) operates without float16 half precision lu_factor errors.
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=1,  # Full batch_size=8 on A100
)

max_epochs = 24
train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=2
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=max_epochs,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="dota/mAP",
        rule="greater",
    )
)
