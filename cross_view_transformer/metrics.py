import torch

from torchmetrics import Metric
from typing import List, Optional

import sys
sys.path.append("/home/jialvzou/cvt_io")
from cross_view_transformer.utils.instance import predict_instance_segmentation_and_trajectories

class BaseIoUMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, thresholds=[0.4, 0.5]):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        thresholds = torch.FloatTensor(thresholds)

        self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
        self.add_state('tp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')

    def update(self, pred, label):
        pred = pred.detach().sigmoid().reshape(-1)
        label = label.detach().bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]

        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)

    def compute(self):
        thresholds = self.thresholds.squeeze(0)
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)

        return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, ious)}


class IoUMetric(BaseIoUMetric):
    def __init__(self, label_indices: List[List[int]], min_visibility: Optional[int] = None):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def update(self, pred, batch):
        if isinstance(pred, dict):
            pred = pred['bev']                                                              # b c h w

        label = batch['bev']                                                                # b n h w
        label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
        label = torch.cat(label, 1)                                                         # b c h w

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            mask = mask[:, None].expand_as(pred)                                            # b c h w

            pred = pred[mask]                                                               # m
            label = label[mask]                                                             # m

        return super().update(pred, label)

class PanopticMetric(Metric):
    def __init__(
            self,
            n_classes: int = 2,
            temporally_consistent: bool = True,
            vehicles_id: int = 1,
            compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.temporally_consistent = temporally_consistent
        self.vehicles_id = vehicles_id
        self.keys = ['iou', 'true_positive', 'false_positive', 'false_negative']

        self.add_state('iou', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')

    def update(self, pred, batch):
        """
        Update state with predictions and targets.

        Parameters
        ----------
            pred_instance: (b, s, h, w)
                Temporally consistent instance segmentation prediction.
            gt_instance: (b, s, h, w)
                Ground truth instance segmentation.
        """

        if self.training:
            return

        # preprocess
        # seg
        pred['segmentation'] = torch.sigmoid(pred['bev'].unsqueeze(1).cpu())
        thresh = 0.5 * torch.ones_like(pred['segmentation'], device=pred['segmentation'].device)
        pred['segmentation'] = torch.cat((thresh, pred['segmentation']), 2)

        # center
        pred['center'] = torch.sigmoid(pred['center'])
        pred['instance_center'] = pred['center'].unsqueeze(1).cpu()
        pred['instance_offset'] = pred['offset'].unsqueeze(1).cpu()

        # no flow
        pred['instance_flow'] = None

        # inference ins
        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            pred, compute_matched_centers=False, make_consistent=True
        )

        # ins gt process
        ins_gt = batch['instance_bev']
        ins_gt_unique = torch.unique(batch['instance_bev'])
        ins_gt_int = torch.zeros_like(batch['instance_bev'], dtype=torch.int64)
        for i in range(len(ins_gt_unique)):
            pos = torch.where(ins_gt == ins_gt_unique[i])
            ins_gt_int[pos] = i

        pred_instance = pred_consistent_instance_seg.cpu()
        gt_instance = ins_gt_int.cpu()

        batch_size, sequence_length = gt_instance.shape[:2]
        # Process labels
        assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        for b in range(batch_size):
            unique_id_mapping = {}
            for t in range(sequence_length):
                result = self.panoptic_metrics(
                    pred_segmentation[b, t].detach(),
                    pred_instance[b, t].detach(),
                    gt_segmentation[b, t],
                    gt_instance[b, t],
                    unique_id_mapping,
                )

                self.iou += result['iou'].to(self.iou.device)
                self.true_positive += result['true_positive'].to(self.true_positive.device)
                self.false_positive += result['false_positive'].to(self.false_positive.device)
                self.false_negative += result['false_negative'].to(self.false_negative.device)

    def update_(self, pred_instance, gt_instance):
        """
        Update state with predictions and targets.

        Parameters
        ----------
            pred_instance: (b, s, h, w)
                Temporally consistent instance segmentation prediction.
            gt_instance: (b, s, h, w)
                Ground truth instance segmentation.
        """
        batch_size, sequence_length = gt_instance.shape[:2]
        # Process labels
        assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        for b in range(batch_size):
            unique_id_mapping = {}
            for t in range(sequence_length):
                result = self.panoptic_metrics(
                    pred_segmentation[b, t].detach(),
                    pred_instance[b, t].detach(),
                    gt_segmentation[b, t],
                    gt_instance[b, t],
                    unique_id_mapping,
                )

                self.iou += result['iou'].to(self.iou.device)
                self.true_positive += result['true_positive'].to(self.true_positive.device)
                self.false_positive += result['false_positive'].to(self.false_positive.device)
                self.false_negative += result['false_negative'].to(self.false_negative.device)

    def compute(self):
        # print("tp: ", self.true_positive)
        # print("fp: ", self.false_positive)
        # print("fn: ", self.false_negative)
        denominator = torch.maximum(
            (self.true_positive + self.false_positive / 2 + self.false_negative / 2),
            torch.ones_like(self.true_positive)
        )
        pq = self.iou / denominator
        sq = self.iou / torch.maximum(self.true_positive, torch.ones_like(self.true_positive))
        rq = self.true_positive / denominator

        return {'pq': pq[1],
                'sq': sq[1],
                'rq': rq[1],
                #  If 0, it means there wasn't any detection.
                'denominator': (self.true_positive + self.false_positive / 2 + self.false_negative / 2)[1],
                }

    def panoptic_metrics(self, pred_segmentation, pred_instance, gt_segmentation, gt_instance, unique_id_mapping):
        """
        Computes panoptic quality metric components.

        Parameters
        ----------
            pred_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
            pred_instance: [H, W] range {0, ..., n_instances} (zero means background)
            gt_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
            gt_instance: [H, W] range {0, ..., n_instances} (zero means background)
            unique_id_mapping: instance id mapping to check consistency
        """
        n_classes = self.n_classes

        result = {key: torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device) for key in self.keys}

        assert pred_segmentation.dim() == 2
        assert pred_segmentation.shape == pred_instance.shape == gt_segmentation.shape == gt_instance.shape

        n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())
        n_all_things = n_instances + n_classes  # Classes + instances.
        n_things_and_void = n_all_things + 1

        #  Now 1 is background; 0 is void (not used). 2 is vehicle semantic class but since it overlaps with
        # instances, it is not present.
        # and the rest are instance ids starting from 3
        prediction, pred_to_cls = self.combine_mask(pred_segmentation, pred_instance, n_classes, n_all_things)
        # print("prediction", prediction.shape)
        # print("pred_to_cls", pred_to_cls.shape)
        target, target_to_cls = self.combine_mask(gt_segmentation, gt_instance, n_classes, n_all_things)

        # Compute ious between all stuff and things
        # hack for bincounting 2 arrays together
        x = prediction + n_things_and_void * target
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void ** 2)
        if bincount_2d.shape[0] != n_things_and_void ** 2:
            raise ValueError('Incorrect bincount size.')
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        # Drop void class
        conf = conf[1:, 1:]

        # Confusion matrix contains intersections between all combinations of classes
        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(union > 0, (conf.float() + 1e-9) / (union.float() + 1e-9),
                          torch.zeros_like(union).float())
        # print("iou: ", iou.shape)

        # In the iou matrix, first dimension is target idx, second dimension is pred idx.
        # Mapping will contain a tuple that maps prediction idx to target idx for segments matched by iou.
        mapping = (iou > 0.5).nonzero(as_tuple=False)

        # Check that classes match.
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching]
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        # First ids correspond to "stuff" i.e. semantic seg.
        # Instance ids are offset accordingly
        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]

            if self.temporally_consistent and cls_id == self.vehicles_id:
                if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                    # Not temporally consistent
                    result['false_negative'][target_to_cls[target_id]] += 1
                    result['false_positive'][pred_to_cls[pred_id]] += 1
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue

            result['true_positive'][cls_id] += 1
            result['iou'][cls_id] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

        for target_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[target_id, n_classes:].any():
                continue
            # If this target instance didn't match with any predictions and was present set it as false negative.
            if target_to_cls[target_id] != -1:
                result['false_negative'][target_to_cls[target_id]] += 1

        for pred_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[n_classes:, pred_id].any():
                continue
            # If this predicted instance didn't match with any prediction, set that predictions as false positive.
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result['false_positive'][pred_to_cls[pred_id]] += 1

        return result

    def combine_mask(self, segmentation: torch.Tensor, instance: torch.Tensor, n_classes: int, n_all_things: int):
        """Shifts all things ids by num_classes and combines things and stuff into a single mask

        Returns a combined mask + a mapping from id to segmentation class.
        """
        instance = instance.view(-1)
        instance_mask = instance > 0
        instance = instance - 1 + n_classes  # Instance ids start from n_classes

        segmentation = segmentation.clone().view(-1)
        segmentation_mask = segmentation < n_classes  # Remove void pixels. background is also masked.

        # Build an index from instance id to class id.
        instance_id_to_class_tuples = torch.cat(
            (
                instance[instance_mask & segmentation_mask].unsqueeze(1),
                segmentation[instance_mask & segmentation_mask].unsqueeze(1),
            ),
            dim=1,
        )
        instance_id_to_class = -instance_id_to_class_tuples.new_ones((n_all_things,))
        # print("instance_id_to_class_tuples: ")
        # print(instance_id_to_class_tuples)
        instance_id_to_class[instance_id_to_class_tuples[:, 0]] = instance_id_to_class_tuples[:, 1]
        instance_id_to_class[torch.arange(n_classes, device=segmentation.device)] = torch.arange(
            n_classes, device=segmentation.device
        )

        segmentation[instance_mask] = instance[instance_mask]
        segmentation += 1  # Shift all legit classes by 1.
        segmentation[~segmentation_mask] = 0  # Shift void class to zero.

        return segmentation, instance_id_to_class
