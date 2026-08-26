"""Registry of models included in the benchmark and their prediction files.

Each entry maps a model key to the COCO-format detections JSON expected in
predictions/. run_eval.py reads this registry to know which prediction files
to evaluate and how to label them in the results table.
"""

GT_ANNOTATION_FILE = "./datasets/dataset_sam_3.0/annotations/instances_test.json"

MODELS = {
    "gccnet": {
        "display_name": "GCC-Net",
        "predictions": "predictions/gccnet.json",
    },
    "retinanet": {
        "display_name": "RetinaNet",
        "predictions": "predictions/retinanet.json",
    },
    "faster_rcnn_fpn": {
        "display_name": "Faster R-CNN+FPN",
        "predictions": "predictions/faster_rcnn_fpn.json",
    },
    "boosting_rcnn": {
        "display_name": "Boosting R-CNN",
        "predictions": "predictions/boosting_rcnn.json",
    },
    "yolov12_x": {
        "display_name": "YOLOv12-X",
        "predictions": "predictions/yolov12_x.json",
    },
    "yolov12_s": {
        "display_name": "YOLOv12-S",
        "predictions": "predictions/yolov12_s.json",
    },
    "dfine_x": {
        "display_name": "D-FINE-X",
        "predictions": "predictions/dfine_x.json",
    },
    "dfine_s": {
        "display_name": "D-FINE-S",
        "predictions": "predictions/dfine_s.json",
    },
}

# Side experiments: NOT part of the main benchmark table (run_eval.py's
# default run only evaluates MODELS above). These use pretraining or setup
# that would be unfair to compare directly against the other models -- e.g.
# boosting_rcnn_utdac starts from the authors' own UTDAC2020-pretrained
# checkpoint, unlike the main boosting_rcnn entry (deliberately ImageNet-only
# init, see project memory for why). Evaluate explicitly with
# `--models boosting_rcnn_utdac` to keep it out of the official table.
EXPERIMENTS = {
    "boosting_rcnn_utdac": {
        "display_name": "Boosting R-CNN (UTDAC2020-pretrained)",
        "predictions": "predictions/boosting_rcnn_utdac.json",
    },
}

# Video-diversity ablation (Experiment 1): same Boosting R-CNN architecture
# and checkpoint-init policy as the main entry, only the training set's video
# coverage changes (50%/75% of videos, two seeds each). Also not part of the
# main benchmark table -- evaluate explicitly with `--models <key>`.
VIDEO_ABLATION = {
    "boosting_rcnn_ablation_50pct_seed42": {
        "display_name": "Boosting R-CNN (50% videos, seed42)",
        "predictions": "predictions/boosting_rcnn_ablation_50pct_seed42.json",
    },
    "boosting_rcnn_ablation_50pct_seed1": {
        "display_name": "Boosting R-CNN (50% videos, seed1)",
        "predictions": "predictions/boosting_rcnn_ablation_50pct_seed1.json",
    },
    "boosting_rcnn_ablation_75pct_seed42": {
        "display_name": "Boosting R-CNN (75% videos, seed42)",
        "predictions": "predictions/boosting_rcnn_ablation_75pct_seed42.json",
    },
    "boosting_rcnn_ablation_75pct_seed1": {
        "display_name": "Boosting R-CNN (75% videos, seed1)",
        "predictions": "predictions/boosting_rcnn_ablation_75pct_seed1.json",
    },
    "dfine_x_ablation_50pct_seed42": {
        "display_name": "D-FINE-X (50% videos, seed42)",
        "predictions": "predictions/dfine_x_ablation_50pct_seed42.json",
    },
    "dfine_x_ablation_50pct_seed1": {
        "display_name": "D-FINE-X (50% videos, seed1)",
        "predictions": "predictions/dfine_x_ablation_50pct_seed1.json",
    },
    "dfine_x_ablation_75pct_seed42": {
        "display_name": "D-FINE-X (75% videos, seed42)",
        "predictions": "predictions/dfine_x_ablation_75pct_seed42.json",
    },
    "dfine_x_ablation_75pct_seed1": {
        "display_name": "D-FINE-X (75% videos, seed1)",
        "predictions": "predictions/dfine_x_ablation_75pct_seed1.json",
    },
}

# Frame-density ablation (Experiment 2): same Boosting R-CNN architecture and
# checkpoint-init policy as the main entry, only the training set's frame
# density changes (N=2 uniform stride, all species, see
# project_frame_selection_ablation memory). Separate key/prediction file from
# the main boosting_rcnn entry -- never overwrites it.
FRAME_ABLATION = {
    "boosting_rcnn_ablation_frame_n2": {
        "display_name": "Boosting R-CNN (frame N=2, all species)",
        "predictions": "predictions/boosting_rcnn_ablation_frame_n2.json",
    },
    "boosting_rcnn_ablation_frame_n3": {
        "display_name": "Boosting R-CNN (frame N=3, all species)",
        "predictions": "predictions/boosting_rcnn_ablation_frame_n3.json",
    },
    "boosting_rcnn_ablation_frame_n4": {
        "display_name": "Boosting R-CNN (frame N=4, all species)",
        "predictions": "predictions/boosting_rcnn_ablation_frame_n4.json",
    },
    "dfine_x_ablation_frame_n2": {
        "display_name": "D-FINE-X (frame N=2, all species)",
        "predictions": "predictions/dfine_x_ablation_frame_n2.json",
    },
    "dfine_x_ablation_frame_n3": {
        "display_name": "D-FINE-X (frame N=3, all species)",
        "predictions": "predictions/dfine_x_ablation_frame_n3.json",
    },
    "dfine_x_ablation_frame_n4": {
        "display_name": "D-FINE-X (frame N=4, all species)",
        "predictions": "predictions/dfine_x_ablation_frame_n4.json",
    },
}

ALL_MODELS = {**MODELS, **EXPERIMENTS, **VIDEO_ABLATION, **FRAME_ABLATION}
