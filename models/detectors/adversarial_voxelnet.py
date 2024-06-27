from typing import Tuple
import torch
from torch import Tensor
from mmdet3d.models.detectors.voxelnet import VoxelNet
from mmdet3d.registry import MODELS
from .. import builder

@MODELS.register_module()
class AdversarialVoxelNet(VoxelNet):
    def __init__(self, adversary_cfg, **kwargs):
        super(AdversarialVoxelNet, self).__init__(**kwargs)
        self.adversary = builder.build_adversary(adversary_cfg)

    def extract_feat(self, batch_inputs_dict: dict, return_perturbations=False) -> Tuple[Tensor, Tensor]:
        """
        Extract features from points, similar to VoxelNet but with adversarial perturbations.
        Optionally return the L2 norm of the perturbations for regularization.
        """
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])

        if hasattr(self, 'adversary'):
            perturbations, l2_norm = self.adversary(voxel_features)
            voxel_features += perturbations
        else:
            perturbations = torch.zeros_like(voxel_features)
            l2_norm = torch.tensor(0.0).to(voxel_features.device)

        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
        x = self.backbone(x)
        
        if self.with_neck:
            x = self.neck(x)

        if return_perturbations:
            return x, l2_norm
        
        return x

    @property
    def perturber(self):
        """Access the adversary module, used for logging and additional operations."""
        return self.adversary

# from typing import Tuple
# import torch
# from torch import Tensor
# from mmdet3d.models.detectors.voxelnet import VoxelNet
# from mmdet3d.registry import MODELS
# from .. import builder

# @MODELS.register_module()
# class AdversarialVoxelNet(VoxelNet):
#     def __init__(self, adversary_cfg, **kwargs):
#         super(AdversarialVoxelNet, self).__init__(**kwargs)
#         self.adversary = builder.build_adversary(adversary_cfg)
    
#     def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor, Tensor]:
#         """Extract features from points, similar to VoxelNet but with adversarial perturbations."""
#         voxel_dict = batch_inputs_dict['voxels']
#         voxel_features = self.voxel_encoder(voxel_dict['voxels'],
#                                             voxel_dict['num_points'],
#                                             voxel_dict['coors'])
#         if hasattr(self, 'adversary'):
#             perturbation = self.adversary(voxel_features)
#             voxel_features += perturbation
#         else:
#             perturbation = torch.zeros_like(voxel_features)

#         batch_size = voxel_dict['coors'][-1, 0].item() + 1
#         x = self.middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
#         x = self.backbone(x)
#         if self.with_neck:
#             x = self.neck(x)
        
#         return x

#     @property
#     def perturber(self):
#         """Property to access the adversary module, specifically for operations like logging."""
#         return self.adversary

# from typing import Tuple
# import torch
# from torch import Tensor
# from mmdet3d.models.detectors.voxelnet import VoxelNet
# from mmdet3d.registry import MODELS
# from .. import builder

# @MODELS.register_module()
# class AdversarialVoxelNet(VoxelNet):
#     def __init__(self, adversary_cfg, **kwargs):
#         super(AdversarialVoxelNet, self).__init__(**kwargs)
#         self.adversary = builder.build_adversary(adversary_cfg)

#     # def extract_feat(self, points, img_metas=None):
#     #     voxels, num_points, coors = self.voxelize(points)
#     #     voxel_features = self.voxel_encoder(voxels, num_points, coors)
#     #     perturbation = self.adversary(voxel_features)
#     #     voxel_features = voxel_features + perturbation
#     #     batch_size = coors[-1, 0].item() + 1
#     #     x = self.middle_encoder(voxel_features, coors, batch_size)
#     #     x = self.backbone(x)
#     #     if self.with_neck:
#     #         x = self.neck(x)
#     #     return x, perturbation
#     def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor, Tensor]:
#         """Extract features from points, similar to VoxelNet but with adversarial perturbations."""
#         voxel_dict = batch_inputs_dict['voxels']
#         voxel_features = self.voxel_encoder(voxel_dict['voxels'],
#                                             voxel_dict['num_points'],
#                                             voxel_dict['coors'])
#         if hasattr(self, 'adversary'):
#             perturbation = self.adversary(voxel_features)
#             voxel_features = voxel_features + perturbation
#         else:
#             perturbation = torch.zeros_like(voxel_features)

#         batch_size = voxel_dict['coors'][-1, 0].item() + 1
#         x = self.middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
#         x = self.backbone(x)
#         if self.with_neck:
#             x = self.neck(x)
        
#         return x#, perturbation

#     # def simple_test(self, points, img_metas, imgs=None, rescale=False):
#     #     x, _ = self.extract_feat(points, img_metas)
#     #     outs = self.bbox_head(x)
#     #     bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
#     #     bbox_results = [
#     #         bbox3d2result(bboxes, scores, labels)
#     #         for bboxes, scores, labels in bbox_list
#     #     ]
#     #     return bbox_results

#     # def aug_test(self, points, img_metas, imgs=None, rescale=False):
#     #     feats, _ = self.extract_feats(points, img_metas)
#     #     aug_bboxes = []
#     #     for x, img_meta in zip(feats, img_metas):
#     #         outs = self.bbox_head(x)
#     #         bbox_list = self.bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
#     #         aug_bboxes.append(bbox_list[0])  # Keep original structure
#     #     merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.bbox_head.test_cfg)
#     #     return [merged_bboxes]

#     # def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore=None):
#     #     x, perturbation = self.extract_feat(points, img_metas)
#     #     outs = self.bbox_head(x)
#     #     loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
#     #     losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
#     #     losses["perturbation_norm"] = torch.mean(torch.linalg.vector_norm(perturbation, dim=1, ord=2))
#     #     losses["perturbation_bias"] = torch.linalg.vector_norm(torch.mean(perturbation, dim=0), ord=2)
#     #     losses["perturbation_imbalance"] = torch.std(torch.mean(perturbation, dim=0))
#     #     return losses, perturbation
