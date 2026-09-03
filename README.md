# Explainable Deep Learning for *Operando* X-ray Diffraction Analysis of Lithium-ion Batteries

**Hamidreza Farhadi Tolie**, **Ashok S. Menon**, **Louis F. J. Piper**, **James Marco**, **Mona Faraji Niri**

![Project](https://img.shields.io/badge/Operando%20XRD%20SoC-Dataset%20%26%20Code-blue)

State of charge is normally inferred from electrical signals that reveal nothing about what is happening inside the electrodes. This repository maps raw two-dimensional *operando* X-ray diffraction frames directly to state of charge with convolutional and sequence models, and explains those predictions quantitatively in reciprocal space.

> Two-dimensional detector frames acquired during cycling are mapped to state of charge without any intermediate reduction to one-dimensional patterns or peak fitting. Models are evaluated under a leave-one-cycle-out protocol so that no frame from an evaluation cycle contributes to training, and predictions are interpreted by projecting Grad-CAM attributions onto the diffraction angle, which shows that the network tracks the reflections that move with lithiation. The framework is transferred to a cell aged over 100 cycles, where prediction fails without adaptation and is recovered by fine-tuning on a small fraction of aged data.

---

## 📁 Repository Structure

```
operando-xrd-soc/
├── Data/
│   ├── fresh/
│   │   ├── 2D/               1536 detector frames (.edf), 487 x 195 pixels
│   │   ├── 1D/               1536 azimuthally averaged patterns (.dat)
│   │   └── metadata.xlsx     filename, timestamp, SoC, C-rate, cycle number
│   └── aged/
│       ├── 2D/               604 detector frames (.edf) after 100 cycles
│       └── metadata.xlsx
│   (the metadata files are included here; the diffraction frames are
│    distributed separately, see Data Availability below)
├── src/
│   ├── core.py                    data loading, splits, models, training, Grad-CAM
│   ├── benchmark_backbones.py     convolutional backbone comparison
│   ├── benchmark_sequence.py      CNN-LSTM sequence models
│   ├── benchmark_multitask.py     joint SoC and charge/discharge prediction
│   ├── baselines_classical.py     PCA, PLS, HOG and peak-feature baselines
│   ├── transfer_aged.py           fresh-to-aged transfer without adaptation
│   └── finetune_aged.py           fine-tuning on the aged cell
├── evaluation/
│   ├── gradcam_quantification.py  attention projected onto 2-theta
│   ├── gradcam_multitask.py       attribution for both prediction heads
│   ├── finetune_fractions.py      learning curve and layer-freezing ablation
│   ├── classification_metrics.py  charge/discharge classification analysis
│   ├── loss_curves.py             training and test convergence
│   └── make_figures.py            publication figures from saved results
├── results/                       created on first run
├── requirements.txt
└── README.md
```

---

## 📡 Data Availability

The `metadata.xlsx` files are included in this repository. The diffraction frames
(1536 fresh and 604 aged `.edf` files, together with the 1536 azimuthally averaged
`.dat` patterns, 842 MB in total) are distributed through Mendeley Data:

> **DOI:** *to be added on publication*

Download the archive and extract it so that the directory layout matches the structure
above, that is `Data/fresh/2D/`, `Data/fresh/1D/` and `Data/aged/2D/` alongside the
metadata files already present. All scripts resolve paths relative to the repository
root and require no further configuration.

---

## 📊 Data Description

Two cells measured at a synchrotron beamline with a Dectris Pilatus 100K detector (0.172 mm pixel pitch, 487 × 195 pixels, 2θ from 16.23° to 42.18°).

**Fresh cell** — 1536 frames over eight complete charge–discharge cycles at C/3, C/3, C/2, C/2, C/1, C/1, C/3, C/3. Both the 2D frames and their azimuthally averaged 1D patterns are provided.

**Aged cell** — 604 frames over six complete cycles at C/3, C/3, C/3, C/2, C/2, C/3, recorded after 100 cycles. An earlier interrupted cycle is excluded.

Each `metadata.xlsx` pairs every frame with its filename, acquisition timestamp, coulomb-counted state of charge, C-rate and cycle number. Frames are matched to metadata **by filename**, not by position. The `SoC_Coulomb` column is expressed as a percentage of the 78 mAh rated capacity, so a fully charged fresh cell reads 107.1 and a fully charged aged cell reads 81.4, corresponding to a measured capacity fade of 24%.

Preprocessing applied identically to every frame: read with `fabio`, rescale to [0, 255] using the frame's own minimum and maximum, resample to 224 × 224 with area interpolation, convert to a float tensor divided by 255. No background subtraction, masking, flat-field correction or conversion to reciprocal space is applied.

---

## 🚀 Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Every script writes CSV results into `results/<name>/` and takes `--help`.

**1. Compare convolutional backbones on the fresh cell**

```bash
python src/benchmark_backbones.py --models ResNet18 DenseNet121 VGG16 EfficientNetB0 MobileNetV2
```

**2. Sequence models over consecutive frames**

```bash
python src/benchmark_sequence.py --seq-len 5
```

**3. Joint state of charge and charge/discharge prediction**

```bash
python src/benchmark_multitask.py
```

**4. Classical baselines**

```bash
python src/baselines_classical.py
```

**5. Transfer to the aged cell, and the base model used for fine-tuning**

```bash
python src/transfer_aged.py --save-base
```

**6. Fine-tune on the aged cell**

```bash
python src/finetune_aged.py --base results/transfer/base_fresh.pth --fraction 0.15 --save-models
```

**7. Quantitative Grad-CAM in reciprocal space**

```bash
python evaluation/gradcam_quantification.py --aged-models results/finetune/finetuned_cycle*.pth
```

**8. Fine-tuning learning curve and layer-freezing ablation**

```bash
python evaluation/finetune_fractions.py --base results/transfer/base_fresh.pth
```

**9. Charge/discharge classification analysis**

```bash
python evaluation/classification_metrics.py
```

**10. Convergence curves and figures**

```bash
python evaluation/loss_curves.py
python evaluation/make_figures.py
```

---

## 🧠 Key Features

- **Direct 2D input** — detector frames are consumed as single-channel images without peak fitting or azimuthal reduction.
- **Cycle-wise validation** — leave-one-cycle-out cross-validation prevents temporally adjacent frames from appearing in both training and evaluation, with `Grouped K-fold`, `Leave-one-C-rate-out` and `Random K-fold` also available for comparison.
- **Sequence modelling** — a causal window of consecutive frames is encoded frame-by-frame and passed to an LSTM, exploiting the temporal correlation of *operando* measurement.
- **Quantitative interpretability** — Grad-CAM attributions are projected onto the diffraction angle and summarised by attention on individual reflections, an enrichment factor relative to uniform attention, and the attention-weighted mean angle.
- **Transfer and adaptation** — the fresh-trained model is evaluated on an aged cell and recovered by fine-tuning on a small fraction of aged frames, with a learning curve and layer-freezing ablation.
- **Error decomposition** — all results are resolved by cycle, C-rate and state-of-charge interval, and reported as mean ± standard deviation across folds.

---

## 📌 Citation

```bibtex
@article{farhaditolie2026operandoxrd,
  title   = {Explainable Deep Learning for Operando X-ray Diffraction Analysis of Lithium-ion Batteries},
  author  = {Farhadi Tolie, Hamidreza and Menon, Ashok S. and Piper, Louis F. J. and Marco, James and Faraji Niri, Mona},
  journal = {},
  year    = {2026}
}
```

---

## 💬 Feedback & Contact

hamidreza.farhadi-tolie@warwick.ac.uk
h.farhaditolie@gmail.com
---

## 🙏 Acknowledgements

WMG, University of Warwick, and The Faraday Institution.

---

## 📄 Licence

Released under the [MIT Licence](LICENSE).

