# Decision log

Each entry has the date, the decision, the alternatives I considered, why I chose this one, and what
happened. All tuning and choices use the validation split. The test split was evaluated once, at the end.

---

## 1. Environment: a project virtual environment with a CUDA 12.8 PyTorch build
**Date:** 2026-11-05

**Decision:** Create `.venv` in the project and install `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128`.
Versions are pinned in `requirements.txt`.

**Alternatives:** Install into the base conda Python. Use an older CUDA 11.8 or 12.1 build.

**Why:** The RTX 5070 is a Blackwell GPU (compute capability sm_120). Older PyTorch builds do not include
kernels for it, so they either fail or silently fall back to the CPU. The cu128 build lists `sm_120` in
`torch.cuda.get_arch_list()`, which I checked. A separate environment keeps the project reproducible and does
not touch other projects.

**Result:** CUDA works, a test matrix multiply runs on the GPU.

---

## 2. Data split: stratified 85/15 train/val from the official train set, official test untouched
**Date:** 2026-11-05

**Decision:** `src/make_splits.py` takes the official train set (8,144 images) and splits it per class into
85% train (6,922) and 15% val (1,222), with seed 42. The official test set (8,041) becomes `test.csv` as is.

**Alternatives:** Use the official test set as validation. Do k-fold cross validation. Use a random split
without stratification.

**Why:** If I tune on the test set, the test score stops being an honest estimate of new data, because I
picked the settings that happen to work on it. So I need my own validation set. Stratifying keeps every class
in val (4 to 10 images each). Without it, some classes could end up with almost no val images. K-fold would
be more precise but would multiply every training run by k, and one fixed split is easier to reason about
and reproduce.

**Result:** train 6,922, val 1,222, test 8,041. All 196 classes are present in all three. Paths are resolved
by scanning the folders, because "Ram C/V Cargo Van Minivan 2012" is stored in a folder named "Ram C-V".
The dataset also unpacks to `archive/car_data/car_data/`, one level deeper than I first wrote down, so the
config points there.

---

## 3. Input size 224 x 224, and squash instead of center crop
**Date:** 2026-11-05

**Decision:** Resize every image (or crop) straight to 224 x 224, even though that changes the aspect ratio.

**Alternatives:** The usual ImageNet recipe: resize the short side to 256, then center crop 224. Pad to a
square (letterbox). Use a larger input like 448, which many fine-grained papers do.

**Why:** In EDA the median car box is 1.9 times wider than tall. A center crop on a wide image cuts off the
front and back of the car, and the headlights, grille, and taillights are often the only things that separate
two similar models. Squashing keeps the whole car. The distortion is the same for every image, so the model
learns it. Padding would waste a large part of the input on gray bars. I kept 224 because that is what the
backbone was pretrained on, and because the robustness experiment uses 224 as the "clean" reference. 448 would
likely add a few points but costs about 4 times the compute per image.

---

## 4. Bounding box crop margin: 10% on each side
**Date:** 2026-11-05

**Decision:** When cropping to the box, add 10% of the box width (left and right) and 10% of the box height
(top and bottom), clipped to the image.

**Alternatives:** No margin (exact box). A fixed pixel margin. A larger margin like 25%.

**Why:** The annotated boxes are tight and sometimes clip a mirror or bumper corner. A relative margin scales
with the car, while a fixed pixel margin would be huge for a small car and nothing for a big one. 10% keeps
the silhouette and leaves room for box errors, which matters later when a YOLO detector, not a human, makes
the boxes. A larger margin brings back the background we are trying to remove.

---

## 5. Train augmentation: mild random crop, horizontal flip, color jitter without hue
**Date:** 2026-11-05

**Decision:**
- Random resized crop keeping 50% to 100% of the area, with the aspect range set *relative to the image's own
  aspect* (3/4 to 4/3 times it), then resize to 224 x 224.
- Horizontal flip with probability 0.5. No vertical flip.
- Color jitter: brightness, contrast, saturation 0.2 each. Hue 0.

**Alternatives:** torchvision defaults (`RandomResizedCrop` with scale 0.08 to 1 and absolute aspect 3/4 to
4/3). Stronger augmentation like RandAugment or Mixup/CutMix. No augmentation.

**Why:** With about 35 images per class, the network memorizes the training set quickly, so some augmentation
is needed. But the default random crop can keep only 8% of the image, which can remove the one detail that
separates two classes. 50% is a safer lower bound. The default also samples crop shapes between 3/4 and 4/3 in
absolute terms. On a 2:1 car photo that means a near-square window that drops half of the car, so I made the
range relative to the image's own shape. A mirrored car is still the same model, so horizontal flip is safe.
Cars are never upside down, so vertical flip would only teach the model things it never needs. Car color is
not part of the label, but large hue shifts create unrealistic colors, so I left hue at 0 and only jitter
brightness, contrast, and saturation. I did not add RandAugment or Mixup to keep the setup simple enough to
explain, and because the results below did not show a large overfitting problem on val.

---

## 6. Backbone: ResNet50 with ImageNet weights (IMAGENET1K_V2), same weights for baseline and fine-tuning
**Date:** 2026-11-05

**Decision:** torchvision ResNet50 with the `IMAGENET1K_V2` weights, for both the frozen baseline and the
fine-tuning runs.

**Alternatives:** The older `IMAGENET1K_V1` weights. EfficientNet or a vision transformer.

**Why:** ResNet50 is a well understood, standard backbone, so it is easy to explain and compare with
published numbers. V2 weights reach 80.9% ImageNet top-1 versus 76.1% for V1. Using the same weights for both
rows of the 2x2 table means the table compares *how* we train, not *what we start from*. EfficientNet was
listed as optional in the plan and I left it out to keep the comparison clean.

---

## 7. Baseline: standardized features + logistic regression, C chosen on val
**Date:** 2026-11-05

**Decision:** Extract the 2048-d pooled feature from the frozen network (eval preprocessing, no
augmentation), standardize each feature using train statistics, and fit multinomial logistic regression.
Choose the regularization `C` from {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0} by val top-1.

**Alternatives:** A fixed `C` = 1 (the sklearn default). A k-NN classifier on the features. An SVM.

**Why:** The sklearn default is not tuned for 2048 features and 35 samples per class. Standardizing first
makes one `C` value apply evenly to all features. My first grid stopped at 0.1 and the best value was on that
edge, which means the true best could be beyond it, so I extended the grid until accuracy clearly flattened.

**Result:** full image: best C = 0.1, val top-1 42.7%. bbox crop: best C = 0.3, val top-1 49.3% (603 of 1,222). Train
accuracy is 99.9% in both, so the linear head memorizes the training set and generic ImageNet features are
the limit. The crop already helps by 6.6 points without any fine-tuning, which shows the gain comes from the
input itself.

---

## 8. Fine-tuning recipe: AdamW, two learning rates, warmup + cosine, bf16, batch 32
**Date:** 2026-11-05

**Decision:**
- Optimizer AdamW, weight decay 0.05 (not applied to biases and BatchNorm weights).
- Head LR = 10 x backbone LR. Backbone LR chosen by a sweep (next entry).
- 1 epoch linear warmup, then cosine decay to 0 over 30 epochs.
- Early stopping: stop if val top-1 has not improved for 6 epochs, keep the best checkpoint.
- Batch size 32. bf16 mixed precision on CUDA. Cross-entropy without label smoothing.

**Why each one:**
- *Two learning rates.* The new 196-class head starts random and needs to move a lot. The backbone is already
  good and a large LR would destroy what it learned on ImageNet. A 10x ratio is a common choice that gives
  the head room to learn while the backbone only adjusts.
- *AdamW and weight decay 0.05.* AdamW works well without much LR tuning. Its weight decay is "decoupled":
  each step shrinks weights by `lr * wd`. With a small LR like 1e-4, the SGD-style value 1e-4 would do almost
  nothing, so I used 0.05, which is in the usual range for AdamW. Biases and normalization weights are
  excluded because shrinking them does not reduce overfitting and can hurt.
- *Warmup.* In the first steps the random head sends large, noisy gradients into the backbone. A short warmup
  keeps those early updates small.
- *Cosine decay.* It lowers the LR smoothly so the last epochs make fine adjustments. It has no extra
  parameters to tune, unlike step decay.
- *30 epochs, patience 6.* The smoke test showed val accuracy already at 74% after 2 epochs, so the model
  converges fast. 30 epochs is enough room, and early stopping on val prevents me from picking the number of
  epochs by hand. Patience 6 is long enough to ride out noise on a small val set (one image is 0.08%).
- *Batch 32.* The dataset is small (6,922 images), so batch 32 still gives 217 updates per epoch. It fits in
  12 GB with room to spare, and 32 is large enough for stable BatchNorm statistics. I did not sweep batch
  size because LR and batch size interact, and sweeping LR at a fixed batch size answers the same question
  with fewer runs.
- *bf16 mixed precision.* About twice as fast on this GPU with no meaningful accuracy loss. bf16 has the same
  range as fp32, so unlike fp16 it does not need loss scaling.
- *No label smoothing.* Label smoothing often adds a little accuracy, but it squeezes the model's confidence
  values together. The confidence threshold study depends on those values meaning something, so I kept plain
  cross-entropy.

**Smoke test:** 2 epochs on bbox crops ran without errors. The first epoch took 60 s (model download, worker
start, cuDNN autotuning), later epochs about 10 s. So a full 30-epoch run is about 5 minutes, which made a
small LR sweep affordable.

---

## 9. Learning rate: sweep {3e-5, 1e-4, 3e-4} for both inputs
**Date:** 2026-11-05

**Decision:** Backbone LR 1e-4 (head 1e-3) for both full image and bbox crop.

**Alternatives:** Use a common default without checking. Pick a separate LR for each input. Take whichever
LR has the single highest val top-1 (that would be 3e-4).

**Why:** The learning rate is the hyperparameter that matters most in fine-tuning, and each run only takes
about 5 minutes, so I tested three values spaced 3x apart around the usual range. Results (best val top-1,
with val NLL in brackets):

| input | 3e-5 | 1e-4 | 3e-4 |
|---|---|---|---|
| full image | 84.5% (0.595) | 86.7% (0.589) | 87.1% (0.593) |
| bbox crop | 88.3% (0.476) | 91.1% (0.427) | 91.2% (0.431) |

3e-5 is clearly too low for both. Between 1e-4 and 3e-4 the gap is 2 images (crop) and 4 images (full) out
of 1,222. The standard error of an accuracy near 91% on 1,222 images is about 0.8 points, so this gap is
noise. When top-1 is a tie, I looked at val NLL (how much probability the model puts on the right answer),
and 1e-4 is better for both inputs. That matters here because the threshold study relies on the
probabilities. 1e-4 is also the gentler choice, it changes the pretrained weights less. I used one LR for
both inputs so the full vs crop comparison is not also a comparison of LRs.

**Result:** The final runs `ft_full` and `ft_crop` are the 1e-4 sweep runs. Re-training them with the same
seed and config would give the same model, so I copied them instead (`copied_from` in their config.json).
Two runs (full 3e-4, crop 1e-4) reached their best val score at the last epoch. With cosine decay that is
expected, because the LR reaches zero at the end and the last epochs are fine adjustments. Val loss was
flat at that point, so I kept 30 epochs.

---

## 10. Model choice from the 2x2 table: fine-tuned ResNet50 on bbox crops
**Date:** 2026-11-05

**Decision:** The best setting on val is fine-tuning on bbox crops (`ft_crop`).

| | full image | bbox crop |
|---|---|---|
| frozen + logistic regression | 42.7% | 49.3% |
| fine-tuned | 86.7% | 91.1% |

**Why:** It has the highest val top-1, and both effects point the same way. Fine-tuning adds about 42 to 44
points, because the backbone learns details that separate similar models, which ImageNet features were never
asked to do. Cropping adds about 4 to 7 points in both rows (6.6 frozen, 4.3 fine-tuned), because the car gets about twice as many pixels
and most of the background, which is mostly unrelated to the car model, is removed.

**Caveat:** At deployment the crop has to come from a detector instead of a human box. So the crop number is
an upper bound until the YOLO step is tested, and a model trained on crops depends on the detector being good.

---

## 11. Degradation augmentation settings
**Date:** 2026-11-05

**Decision:** Retrain `ft_crop` with random degradation added to training (`ft_crop_degaug`). Each of three
corruptions is applied on its own with probability 0.3, after the normal augmentation:
downscale to a random size between 24 and 160 px (then back up to 224), Gaussian blur with sigma 0.5 to 3,
and brightness multiplied by 0.2 to 0.8. Everything else is the same as `ft_crop`.

**Alternatives:** Always degrade every image. Train only on one corruption type. Use the exact test levels
(32, 64, 112 px, and so on) in training.

**Why:** Probability 0.3 per corruption means about a third of the images stay clean, so the model keeps
learning fine details from sharp images. The ranges are continuous and cover the test levels, but are not
the exact test values, so a gain on the curves shows real robustness and not memorization of 4 specific
settings.

**Decision rule, set before seeing the results:** use `ft_crop_degaug` as the final model only if its clean
val top-1 is within 1.0 point of `ft_crop` and it is better on average across the degraded conditions.
Otherwise keep `ft_crop`.

---

## 12. Confidence threshold rule: lowest threshold with at least 95% accuracy on accepted images
**Date:** 2026-11-05

**Decision:** The operating rule is "use the lowest confidence threshold at which val accuracy on accepted
images is at least 95%". Images below the threshold are answered as "unknown" and go to manual review.

**Alternatives:** A fixed value such as 0.5. Maximize some combined score such as F1. Choose a fixed reject
rate, for example "reject 10%".

**Why:** The threshold should come from what the system is for, not from a round number. In an inspection
setting, a confident wrong answer costs more than a "please check" answer, because nobody looks at it again.
So I fix the quality of what the system accepts (19 of 20 accepted answers right) and then reject as little
as possible to get there. The lowest threshold that reaches the target keeps coverage as high as possible.
The 95% target is a business choice. If wrong answers were more costly, the same procedure works with 98% or
99%. Because val is small, I also report a bootstrap 95% confidence interval for coverage and accepted
accuracy at the chosen threshold.

---

## 13. Error analysis: what the confusions say
**Date:** 2026-11-06

**What I looked at:** errors of `ft_crop` on val (109 of 1,222, 8.9%).

**Findings:**
- 57.8% of errors are within the same make. If errors were random among the 195 wrong classes, only 4.1%
  would be. So the model almost always knows the make (96.2% make-level accuracy) and fails on the model,
  body style, or year inside it.
- The top confused pairs are the hard cases I expected from EDA: Audi V8 vs Audi 100 (1994), Dodge Caliber
  2007 vs 2012 (year only), Ferrari 458 coupe vs convertible, Audi 100 sedan vs wagon.
- Most cross-make errors are not really cross-make. They are rebadged twins built on the same body:
  Chevrolet Tahoe vs GMC Yukon, Chevrolet Express vs GMC Savana, Dodge Sprinter vs Mercedes-Benz Sprinter.
  These cars differ mostly in the badge and grille.
- Coupe vs convertible errors (Ferrari, Bugatti, Spyker) happen on photos where the roof is up or not
  visible, so the image honestly does not contain the answer.
- The most confident mistakes have confidence near 1.00. A threshold cannot catch those, so they are a
  limit of the method, not of the threshold value.

**What I would do next:** a higher input resolution (for badges and grille texture), more data for year-only
pairs, and treating rebadged twins as one class if the use case only needs the body type.

---

## 14. Final model: the degradation-augmented crop model, threshold 0.76
**Date:** 2026-11-06

**Decision:** The final model is `ft_crop_degaug` and its operating threshold is 0.76. Both were fixed on val
and saved to `outputs/final_choice.json` before the test split was opened.

**Why, using the rule from entry 11:**

| val top-1 | ft_crop | ft_crop_degaug |
|---|---|---|
| clean | 91.1% | 90.5% |
| 112 px / 64 px / 32 px | 75.7% / 37.1% / 4.3% | 90.3% / 86.8% / 70.3% |
| blur sigma 2 / 4 | 44.8% / 6.5% | 87.9% / 73.3% |
| brightness 0.15 | 84.5% | 89.4% |

The clean cost is 0.6 points (about 7 images, inside the noise), and the average gain over the 10 degraded
conditions is 31.9 points. Without the augmentation, the model collapses at 64 px and at blur sigma 2, which
is what a far-away car on a street camera looks like. Darkness hurt much less than I expected, probably
because brightness jitter was already in the normal augmentation.

**Threshold:** the same 95% rule from entry 12, re-run on the final model's val predictions, gives 0.76.
On val that means 87.6% coverage (95% CI 85.7% to 89.4%) and 95.1% accepted accuracy (95% CI 93.8% to
96.3%). The model is reasonably calibrated (expected calibration error 0.035), so its confidence values
can be read roughly as probabilities. On degraded images, coverage drops (59.5% at 32 px) while accepted
accuracy stays near 88% to 95%. In other words, the model rejects more when the image is worse, which is the
behavior we want.

**Note on small number differences:** the best-epoch val score logged during training (90.6%) and the
re-evaluation (90.5%) differ by one image. GPU kernels chosen by cuDNN autotuning and bf16 arithmetic are not
bit-for-bit deterministic. One image is far below anything I made a decision on.

---

## 15. Test evaluation (once)
**Date:** 2026-11-06

**What I did:** ran `src/final_test.py` once, with the model and threshold frozen in entry 14. Nothing was
changed after seeing these numbers.

**Result on the official test set (8,041 images):**
- top-1 89.5%, top-5 98.4%, mean per-class accuracy 89.5%
- at threshold 0.76: coverage 87.3%, accuracy on accepted 95.2%
- degraded: 64 px 86.0%, 32 px 69.2%, blur sigma 4 73.9%, brightness 0.15 89.5%

**What this tells me:** test top-1 is 1.0 point below val (90.5%), which is the normal small optimism of a
val set that was used for choices. The threshold transferred almost exactly: val promised 87.6% coverage at
95.1% accuracy and test gave 87.3% at 95.2%. That is the main evidence that choosing the threshold on val was
sound.

**Limits:** these are still clean web photos with human-drawn boxes. Real footage needs a detector for the
crop, and many cars on the road today are newer than every class here. The threshold handles some of those as
"unknown", but I did not measure open-set rejection directly because there was no held-out set of unknown
car models.
