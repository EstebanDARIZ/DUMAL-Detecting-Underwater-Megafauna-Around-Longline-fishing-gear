# DUMAL

Code accompanying the paper *DUMAL: An Underwater Megafauna Detection Dataset for Sustainable Longline Fisheries*, submitted to the WACV 2027 Evaluation and Dataset Track.

**Dataset:** [https://dataverse.harvard.edu/previewurl.xhtml?token=2da98596-a101-4bd4-b3b8-28a5ac1f458e]

## Repository structure

### `dataset_creation/`
Pipeline that builds the released dataset from raw per-run annotation output: splits runs into train/test pools, flattens and renames frames, splits train/val, and converts YOLO-format labels to COCO. Also computes the same-camera/same-day video clustering used throughout the paper's statistics and splits.

### `dataset_stats/`
Computes the two dataset-statistics tables reported in the paper (per-split counts of images, instances, source videos, and clusters, and the per-class breakdown by Train+Val vs Test).

### `annotation_quality/`
Computes the two inter-annotator agreement studies reported in the paper: a localization-agreement study based on IoU, and a classification-agreement study based on Cohen's kappa. Includes sample selection, clip extraction, the dual-panel annotation tool used to collect comparison labels, and the agreement computations themselves.

### `annotation_tool/`
The semi-automatic annotation tool used to label the dataset: an annotator draws an initial bounding box on the frame where an animal first appears, and SAM3, used as a promptable video object tracker, propagates the corresponding box across subsequent frames.

### `benchmark/`
Evaluation of the eight detector configurations benchmarked in the paper (D-FINE-X/S, YOLOv12-X/S, RetinaNet, Faster R-CNN+FPN, GCC-Net, Boosting R-CNN), including the pretraining-source comparison and the JellyFish-specific evaluation.

### `ablation/`
The two controlled ablations reported in the paper: reducing the training set by video-cluster subsampling (75%/50% of clusters kept), and by frame-spacing subsampling (every 2nd/3rd/4th frame kept).
