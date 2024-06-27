_base_ = [
    '../../mmdetection3d/configs/_base_/models/second_hv_secfpn_kitti.py',
    '../../mmdetection3d/configs/_base_/datasets/kitti-3d-car.py',
    '../../mmdetection3d/configs/_base_/schedules/cyclic-40e.py',
    '../../mmdetection3d/configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=['models.detectors.adversarial_voxelnet', 'models.adversarial.voxel_perturber'], allow_failed_imports=False)

# default_scope = 'mmdet3d'
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
model = dict(
    type="AdversarialVoxelNet",
    adversary_cfg=dict(
        type='VoxelPerturber',  
     
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False))

### Another attempt
# model = dict(
#     type='AdversarialVoxelNet',
#     data_preprocessor=dict(
#         type='Det3DDataPreprocessor',
#         voxel=True,
#         voxel_layer=dict(
#             max_num_points=5,
#             point_cloud_range=[0, -40, -3, 70.4, 40, 1],
#             voxel_size=[0.05, 0.05, 0.1],
#             max_voxels=(16000, 40000)
#         )
#     ),
#     voxel_encoder=dict(
#         type='HardSimpleVFE',
#         num_features=4,  # Example parameter, adjust based on actual implementation needs
#     ),
#     middle_encoder=dict(
#         type='SparseEncoder',
#         in_channels=4,
#         sparse_shape=[41, 1600, 1408],
#         order=('conv', 'norm', 'act')
#     ),
#     backbone=dict(
#         type='SECOND',
#         in_channels=256,
#         layer_nums=[5, 5],
#         layer_strides=[1, 2],
#         out_channels=[128, 256]
#     ),
#     neck=dict(
#         type='SECONDFPN',
#         in_channels=[128, 256],
#         upsample_strides=[1, 2],
#         out_channels=[256, 256]
#     ),
#     adversary_cfg=dict(
#         type='VoxelPerturber',  # Additional parameters can be specified here if needed
#     ),
#     bbox_head=dict(
#         type='Anchor3DHead',
#         num_classes=1,
#         in_channels=512,
#         feat_channels=512,
#         anchor_generator=dict(
#             type='Anchor3DRangeGenerator',
#             ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
#             sizes=[[3.9, 1.6, 1.56]],  # Sizes of the anchors (width, length, height)
#             rotations=[0, 1.57],  # Rotations of the anchors in radians
#             reshape_out=True
#         ),
#         bbox_coder=dict(
#             type='DeltaXYZWLHRBBoxCoder'
#         ),
#         loss_cls=dict(
#             type='mmdet.FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=1.0
#         ),
#         loss_bbox=dict(
#             type='mmdet.SmoothL1Loss',
#             beta=1.0 / 9.0,
#             loss_weight=2.0
#         ),
#         loss_dir=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=False,
#             loss_weight=0.2
#         )
#     ),
#     train_cfg=dict(
#         assigner=dict(
#             type='Max3DIoUAssigner',
#             iou_calculator=dict(type='BboxOverlapsNearest3D'),
#             pos_iou_thr=0.6,
#             neg_iou_thr=0.45,
#             min_pos_iou=0.45,
#             ignore_iof_thr=-1
#         ),
#         allowed_border=0,
#         pos_weight=-1,
#         debug=False
#     ),
#     test_cfg=dict(
#         use_rotate_nms=True,
#         nms_across_levels=False,
#         nms_thr=0.01,
#         score_thr=0.1,
#         min_bbox_size=0,
#         nms_pre=100,
#         max_num=50
#     )
# )
