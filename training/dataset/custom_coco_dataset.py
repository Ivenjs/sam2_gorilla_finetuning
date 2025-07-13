import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os

from .vos_segment_loader import SegmentLoader
from .vos_raw_dataset import VOSRawDataset, VOSVideo, VOSFrame


class GorillaCOCORawDataset(VOSRawDataset):
    """
    A raw dataset class for a single COCO annotation file.
    This class returns VOSVideo and COCOSegmentLoader objects.
    """
    def __init__(self, img_folder: str, gt_file: str, **kwargs):
        self.img_folder = img_folder
        self.coco = COCO(gt_file)
        # Get a list of all image IDs that have annotations
        self.image_ids = sorted(self.coco.getImgIds())
        
        # We can think of each "image" as a "video" of length 1
        self.video_names = [self.coco.loadImgs(img_id)[0]['file_name'] for img_id in self.image_ids]

    def get_video(self, idx: int):
        """
        Returns a VOSVideo object and a COCOSegmentLoader for the given image index.
        """
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.img_folder, img_info['file_name'])

        # Create the segment loader for this specific image
        segment_loader = COCOSegmentLoader(coco_api=self.coco, image_id=image_id)

        # Create a single frame for our "video"
        frames = [VOSFrame(frame_idx=0, image_path=image_path)]
        
        # Create the VOSVideo object
        video = VOSVideo(
            video_name=os.path.splitext(img_info['file_name'])[0],
            video_id=image_id,
            frames=frames
        )
        
        return video, segment_loader

    def __len__(self):
        return len(self.image_ids)

class COCOSegmentLoader(SegmentLoader):
    """
    A segment loader for a single COCO annotation file.
    """
    def __init__(self, coco_api: COCO, image_id: int):
        self.coco = coco_api
        self.image_id = image_id
        # Get all annotation IDs for this specific image
        self.ann_ids = self.coco.getAnnIds(imgIds=self.image_id)
        self.annotations = self.coco.loadAnns(self.ann_ids)

    def get_mask(self, frame_idx: int, object_id: int) -> torch.Tensor:
        """
        Loads the mask for a given object in the frame.
        frame_idx is ignored since we have a single image.
        object_id is the index into our list of annotations for this image.
        """
        if object_id >= len(self.annotations):
            return None # No such object
        
        ann = self.annotations[object_id]
        mask = self.coco.annToMask(ann)
        return torch.from_numpy(mask)

    @property
    def num_objects(self) -> int:
        return len(self.annotations)