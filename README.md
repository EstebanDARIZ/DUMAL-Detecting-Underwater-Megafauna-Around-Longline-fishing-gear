# DUMAL: Detecting Underwater Megafauna Around Longline fishing gear

Code and dataset for the paper *DUMAL: An Underwater Megafauna Detection Dataset for Sustainable Longline Fisheries*, accepted at the WACV 2027 Datasets Track.

**Dataset:** [https://doi.org/10.7910/DVN/QM19AI](https://doi.org/10.7910/DVN/QM19AI) (Harvard Dataverse) · **Project page:** [https://l3i-share.univ-lr.fr/2026DUMAL](https://l3i-share.univ-lr.fr/2026DUMAL)

Esteban Darizcuren-Dreau<sup>1</sup>, Nicolas Sidere<sup>1</sup>, Jean-Christophe Burie<sup>1</sup>, François Poisson<sup>2</sup>
<sup>1</sup>L3i, La Rochelle Université, France · <sup>2</sup>MARBEC, Univ Montpellier, CNRS, IFREMER, IRD, Montpellier, France

## Abstract

Longline fishing poses significant bycatch risks to protected marine species, yet no public computer-vision dataset records megafauna in situ around active fishing gear. We introduce DUMAL (Detecting Underwater Megafauna Around Longline fishing gear), a dataset of 60,870 annotated frames (94,101 boxes across nine species) from 55 independent videos recorded on an instrumented pelagic longline in the Mediterranean Sea. Two inter-annotator agreement studies validate annotation quality: a localization study (mean IoU 0.605 detection-and-localization, 0.811 localization-only, within four points of the human ceiling) and a classification study (Cohen's κ = 0.86). We split train and test hermetically at the video level, a practice uncommon in underwater detection datasets. Benchmarking eight detector configurations, we find that generalization to unseen footage depends far more on the number of distinct training videos a class appears in than on its instance count, confirmed by two controlled ablations that independently vary video count and frame density.

## Dataset

DUMAL is part of the MEGABIT project. Cameras were mounted on a subset of branch lines of a pelagic longline, facing downward 1–2 m above the baited hook. Footage was recorded between 2019 and 2025 on two Mediterranean fishing grounds (Corsica and the Gulf of Lions, France), by day and by night, during regular commercial fishing operations.

| Split | Images | Instances | Source videos | Clusters |
|-------|-------:|----------:|--------------:|---------:|
| Train | 48,006 | 74,606 | 41 | 25 |
| Val   | 5,333  | 8,358  | 41* | 25* |
| Test  | 7,531  | 11,137 | 14 | 12 |
| **Total** | **60,870** | **94,101** | **55** | **37** |

\* Same source videos and clusters as Train. Videos recorded by the same camera on the same day form a cluster and are always kept in the same split.

**Classes:** Squid, Sardine, Ray, Sunfish, Pilot Fish, Shark, JellyFish, Tuna, Mackerel. JellyFish appears in a single source video, so it has no test instances and is not evaluated.

**Formats:** annotations are provided in COCO JSON (`annotations/instances_<split>.json`, category ids 1–9) and YOLO (`<split>/labels/*.txt`, class ids 0–8, see `data.yaml`). Images come at 1280×720, 1920×1080 or 2560×1440. Frames named `*_bg_*` are background frames with no animal and no annotation. The reduced training sets used in the two ablations are also provided as COCO JSON files.

### Download

The dataset (about 13.7 GB) is hosted on Harvard Dataverse: [https://doi.org/10.7910/DVN/QM19AI](https://doi.org/10.7910/DVN/QM19AI). It contains:

- `DUMAL_annotations.zip`: COCO and YOLO annotations, `data.yaml` and ablation splits (~28 MB)
- `train_images_part001.zip` to `train_images_part005.zip`: 48,006 train images (~11.1 GB)
- `val_images_part001.zip`: 5,333 val images (~1.2 GB)
- `test_images_part001.zip`: 7,531 test images (~1.3 GB)

Download all the files into the same directory, then rebuild the dataset folder:

```bash
unzip DUMAL_annotations.zip
unzip 'train_images_part*.zip' -d DUMAL/train/images
unzip val_images_part001.zip -d DUMAL/val/images
unzip test_images_part001.zip -d DUMAL/test/images
```

This gives the following structure:

```
DUMAL/
├── data.yaml            YOLO configuration and class names
├── annotations/         instances_{train,val,test}.json (COCO)
├── train/{images,labels}/
├── val/{images,labels}/
├── test/{images,labels}/
└── ablations/
    ├── ablation_video/  75% and 50% of video clusters, seeds 1 and 42
    └── ablation_frame/  one frame kept every N frames (N = 2 to 5)
```

Machine-readable metadata are given in [`dumal_croissant.json`](dumal_croissant.json) ([Croissant](http://mlcommons.org/croissant/) format).

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
The two controlled ablations reported in the paper: reducing the training set by video-cluster subsampling (75%/50% of clusters kept), and by frame-spacing subsampling (one frame kept every N frames).

## License

The DUMAL dataset is released under the [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## Citation

If you use DUMAL in your research, please cite:

```bibtex
@inproceedings{darizcuren2027dumal,
  title     = {{DUMAL}: An Underwater Megafauna Detection Dataset for Sustainable Longline Fisheries},
  author    = {Darizcuren-Dreau, Esteban and Sidere, Nicolas and Burie, Jean-Christophe and Poisson, Fran{\c{c}}ois},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Acknowledgments

The at-sea experiments were conducted as part of two research programs: the Gen&Rec project ("Distribution and behaviour of swordfish spawners and juveniles in the vicinity of Corsica"), funded by the European Union (European Maritime and Fisheries Fund, EMFF) and the fishing industry (France Filière Pêche, FFP); and the MOUVBLUE project ("Movements of blue sharks"), funded by the Office Français de la Biodiversité (OFB) and the France Relance plan.

## Contact

Esteban Darizcuren-Dreau: esteban.dreau-darizcuren@univ-lr.fr
