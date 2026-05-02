from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from instance_segmentation.abs_instance_segmenter import (
    AbsInstanceSegmenter,
    SegmentationResults,
)
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from instance_segmentation.point_prompted_segmentation.abs_point_prompted_segmenter import (
    AbsPointPromptedSegmenter,
)
from utils.visualization_utils import overlay_single_mask
from PIL import Image


def _connected_components_8(binary_mask: np.ndarray) -> List[np.ndarray]:
    # OpenCV returns label 0 for background and [1..num_labels-1] for components.
    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )
    components: List[np.ndarray] = []
    for label_idx in range(1, num_labels):
        ys, xs = np.where(labels == label_idx)
        if ys.size == 0:
            continue
        components.append(np.stack([ys, xs], axis=1).astype(np.int32))

    return components


class LanguageGroundedWithPointRepromptSegmenter(AbsInstanceSegmenter):
    """
    Two-stage segmenter:
    1) run a language-grounded segmenter,
    2) sample points in uncovered regions and segment with point prompts.
    """

    def __init__(
        self,
        *,
        language_grounded_segmenter: AbsInstanceSegmenter,
        point_prompted_segmenter: AbsPointPromptedSegmenter,
        min_reprompt_region_area_perc: float = 0.002,
        min_reprompt_region_area_pixels: Optional[int] = None,
        large_region_area_perc_for_grid: float = 0.02,
        grid_stride_percentage: float = 0.075,
        max_points_per_region: int = 16,
        include_region_center_point: bool = True,
        point_label: int = 1,
        min_area_perc: float = 0.0,
        dedupe_iou_threshold: float = 1.01,
        confidence_threshold: Optional[float] = None,
        name_suffix="",
    ):
        super().__init__(
            name=f"{language_grounded_segmenter.getName()}_reprompt_{point_prompted_segmenter.getName()}{name_suffix}",
            min_area_perc=min_area_perc,
            dedupe_iou_threshold=dedupe_iou_threshold,
            confidence_threshold=confidence_threshold,
        )
        self.language_grounded_segmenter = language_grounded_segmenter
        self.point_prompted_segmenter = point_prompted_segmenter

        self.min_reprompt_region_area_perc = float(min_reprompt_region_area_perc)
        self.min_reprompt_region_area_pixels = min_reprompt_region_area_pixels
        self.large_region_area_perc_for_grid = float(large_region_area_perc_for_grid)
        self.grid_stride_percentage = float(grid_stride_percentage)
        self.max_points_per_region = int(max_points_per_region)
        self.include_region_center_point = bool(include_region_center_point)
        self.point_label = int(point_label)

        if self.min_reprompt_region_area_perc < 0:
            raise ValueError("min_reprompt_region_area_perc must be >= 0.")
        if self.min_reprompt_region_area_pixels is not None and self.min_reprompt_region_area_pixels < 0:
            raise ValueError("min_reprompt_region_area_pixels must be >= 0.")
        if self.large_region_area_perc_for_grid < 0:
            raise ValueError("large_region_area_perc_for_grid must be >= 0.")
        if self.grid_stride_percentage <= 0:
            raise ValueError("grid_stride_percentage must be > 0.")
        if self.max_points_per_region <= 0:
            raise ValueError("max_points_per_region must be > 0.")

    @staticmethod
    def _point_in_component_near_centroid(component: np.ndarray) -> Tuple[int, int]:
        cy = float(component[:, 0].mean())
        cx = float(component[:, 1].mean())
        d2 = (component[:, 0] - cy) ** 2 + (component[:, 1] - cx) ** 2
        idx = int(np.argmin(d2))
        y, x = component[idx]
        return int(x), int(y)

    def _sample_points_for_component(
        self,
        component: np.ndarray,
        image_width: int,
        image_height: int,
        *,
        large_region_area_threshold: int,
    ) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        point_set = set()
        area = int(component.shape[0])

        if self.include_region_center_point:
            center = self._point_in_component_near_centroid(component)
            points.append(center)
            point_set.add(center)

        # Calculate the grid stride in pixels (as an average of the width and height percentage)
        grid_stride_px = int(0.5 * (self.grid_stride_percentage * image_width + self.grid_stride_percentage * image_height))

        if area >= large_region_area_threshold:
            ys = component[:, 0]
            xs = component[:, 1]
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            region_points = {(int(x), int(y)) for y, x in component.tolist()}
            y_start = y_min + grid_stride_px // 2
            x_start = x_min + grid_stride_px // 2

            for y in range(y_start, y_max + 1, grid_stride_px):
                for x in range(x_start, x_max + 1, grid_stride_px):
                    candidate = (x, y)
                    if candidate not in region_points or candidate in point_set:
                        continue
                    points.append(candidate)
                    point_set.add(candidate)
                    if len(points) >= self.max_points_per_region:
                        return points

        if len(points) > self.max_points_per_region:
            # Shuffle the points
            random.shuffle(points)
            points = points[: self.max_points_per_region]
        return points

    def segmentInstancesRaw(self, image: torch.Tensor) -> Tuple[SegmentationResults, DebugPayload]:
        _, h, w = image.shape

        first_pass, _first_pass_debug = self.language_grounded_segmenter.segmentInstances(image, image_id="__nested__")
        # print(f"Number of masks after language grounded segmentation: {len(first_pass)}")



        if len(first_pass) == 0:
            covered = torch.zeros((h, w), dtype=torch.bool)
        else:
            covered = first_pass.masks.any(dim=0)
        uncovered = ~covered

        # # Display the uncovered pixels on the image as a mask and save to a file
        # # Use the overlay function from utils.image_utils
        # overlayed_image = overlay_single_mask(image, uncovered)
        # # Convert the image to a PIL image and save to file
        # overlayed_image_pil = Image.fromarray(overlayed_image)
        # overlayed_image_pil.save("debug_outputs/uncovered_pixels.png")


        # circle_radius = max(1, int(0.01 * w))


        # Print the percent of uncovered pixels
        # print(f"Percent of uncovered pixels: {uncovered.sum() / (h * w)}")

        if not bool(uncovered.any()):
            return first_pass, DebugPayload.empty()

        uncovered_np = uncovered.cpu().numpy().astype(bool)
        components = _connected_components_8(uncovered_np)

        min_region_area = (
            int(self.min_reprompt_region_area_pixels)
            if self.min_reprompt_region_area_pixels is not None
            else int(self.min_reprompt_region_area_perc * h * w)
        )
        large_region_area_threshold = int(self.large_region_area_perc_for_grid * h * w)

        sampled_points: List[Tuple[int, int]] = []
        for i, component in enumerate(components):
            if component.shape[0] < min_region_area:
                continue

            # Convert the component to a mask
            # component_mask = np.zeros((h, w), dtype=np.uint8)
            # component_mask[component[:, 0], component[:, 1]] = 255
            # component_mask_tensor = torch.from_numpy(component_mask)
            # SAve the component as a mask and save to a file
            # overlayed_image = overlay_single_mask(image, component_mask_tensor)
            # overlayed_image_pil = Image.fromarray(overlayed_image)
            # overlayed_image_pil.save(f"debug_outputs/component_{i}_mask.png")

            pointsForComponent = self._sample_points_for_component(
                component,
                image_width=w,
                image_height=h,
                large_region_area_threshold=large_region_area_threshold,
            )

            # Display the points for the component on the overlayed image and save to a file
            # for point in pointsForComponent:
            #     x, y = point
            #     cv2.circle(overlayed_image, (x, y), circle_radius, (0, 0, 255), -1)
            # overlayed_image_pil = Image.fromarray(overlayed_image)
            # overlayed_image_pil.save(f"debug_outputs/component_{i}_points.png")

            sampled_points.extend(pointsForComponent)

        if len(sampled_points) == 0:
            return first_pass, DebugPayload.empty()

        # Display sampled points on a contiguous HWC uint8 image for OpenCV drawing.
        # If the image is a tensor, detach and convert to numpy
        # if isinstance(image, torch.Tensor):
        #     sampled_points_image = image.detach().cpu()
        #     if sampled_points_image.ndim == 3 and sampled_points_image.shape[0] in (1, 3):
        #         sampled_points_image = sampled_points_image.permute(1, 2, 0)
        #     sampled_points_image = sampled_points_image.numpy()
        # else:
        #     sampled_points_image = np.ascontiguousarray(image)
        # if sampled_points_image.max() <= 1.0:
        #     sampled_points_image = (np.clip(sampled_points_image, 0, 1) * 255).astype(np.uint8)
        # else:
        #     sampled_points_image = np.clip(sampled_points_image, 0, 255).astype(np.uint8)
        # sampled_points_image = np.ascontiguousarray(sampled_points_image)


        # for point in sampled_points:
        #     x, y = point
        #     cv2.circle(sampled_points_image, (x, y), circle_radius, (0, 0, 255), -1)
        # Convert the image to a PIL image and save to file
        # sampled_points_image_pil = Image.fromarray(sampled_points_image)
        # sampled_points_image_pil.save("debug_outputs/sampled_points.png")


        point_coords = torch.tensor(sampled_points, dtype=torch.float32)
        point_labels = torch.full(
            (point_coords.shape[0],),
            fill_value=self.point_label,
            dtype=torch.int64,
        )
        second_pass = self.point_prompted_segmenter.segment_from_points(
            image=image,
            point_coords=point_coords,
            point_labels=point_labels,
        )

        # print(f"Number of masks after point prompted segmentation: {len(second_pass)}")

        if second_pass.masks.shape[0] == 0:
            return first_pass, DebugPayload.empty()

        second_seg = SegmentationResults(
            ids=torch.arange(second_pass.masks.shape[0], dtype=torch.int64),
            masks=second_pass.masks.to(torch.bool),
            boxes=second_pass.boxes.to(torch.float32),
            scores=second_pass.scores.to(torch.float32),
        )

        if len(first_pass) == 0:
            return SegmentationResults(
                ids=torch.arange(len(second_seg), dtype=torch.int64),
                masks=second_seg.masks,
                boxes=second_seg.boxes,
                scores=second_seg.scores,
            ), DebugPayload.empty()

        first_scores = (
            first_pass.scores.to(torch.float32)
            if first_pass.scores is not None
            else torch.zeros((len(first_pass),), dtype=torch.float32)
        )
        second_scores = (
            second_seg.scores.to(torch.float32)
            if second_seg.scores is not None
            else torch.zeros((len(second_seg),), dtype=torch.float32)
        )

        merged_masks = torch.cat([first_pass.masks.to(torch.bool), second_seg.masks.to(torch.bool)], dim=0)
        merged_boxes = torch.cat([first_pass.boxes.to(torch.float32), second_seg.boxes.to(torch.float32)], dim=0)
        merged_scores = torch.cat([first_scores, second_scores], dim=0)
        merged_ids = torch.arange(merged_masks.shape[0], dtype=torch.int64)

        # Print the total number of masks after merging
        # print(f"Total number of masks after merging: {len(merged_masks)}")

        return SegmentationResults(
            ids=merged_ids,
            masks=merged_masks,
            boxes=merged_boxes,
            scores=merged_scores,
        ), DebugPayload.empty()

