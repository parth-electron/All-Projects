# Face Recognition "in the Wild" — CNN on the LFW Dataset



## Objective

Build a Convolutional Neural Network (CNN) that recognizes a person's identity from a cropped
face photo taken under real-world ("in the wild") conditions — varied pose, lighting, and
background — and evaluate how well the model performs.

## Dataset

Labeled Faces in the Wild (LFW):
http://vis-www.cs.umass.edu/lfw/
(also on Kaggle: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

No manual download is required — the notebook loads it directly via
`sklearn.datasets.fetch_lfw_people`, which downloads and caches the dataset automatically the
first time it runs (requires internet access on that first run). The `min_faces_per_person=70`
filter keeps only identities with enough photos to make classification meaningful, keeping this a
tractable classroom-scale problem.

## Libraries Used

- `scikit-learn` — `fetch_lfw_people` for the dataset, `train_test_split`, classification report
  and confusion matrix utilities
- `tensorflow` / `tensorflow.keras` — building and training the CNN
- `numpy` — numerical operations
- `matplotlib` — plotting sample faces, training curves, and the confusion matrix

## Methodology

1. **Data Understanding** — loaded LFW via `fetch_lfw_people` (grayscale, resized to 50% for a
   smaller/faster CNN), inspected the number of samples, image dimensions, and identity classes,
   and displayed sample faces with their labels.
2. **Data Preprocessing**
   - Verified there are no missing/corrupt values.
   - Normalized pixel values from `[0, 255]` to `[0, 1]`.
   - Added the channel dimension the CNN expects and one-hot encoded the identity labels.
   - Split the data into 80% training / 20% testing, stratified on identity so every person is
     proportionally represented in both sets.
3. **Model Development** — built a CNN with three convolutional blocks (Conv2D + Batch
   Normalization + MaxPooling + Dropout, increasing from 32 to 128 filters) followed by a dense
   classification head, trained for 30 epochs with the Adam optimizer.
4. **Model Evaluation** — computed test loss/accuracy, a full classification report
   (precision/recall/F1 per identity), plotted training/validation accuracy and loss curves, and
   generated a confusion matrix across identities.

## Results


| Metric | Value |
|---|---|
| Test Accuracy | *(fill in after running)* |
| Test Loss | *(fill in after running)* |

**Observations **


## Conclusion

This project built a Convolutional Neural Network to recognize faces "in the wild" using the LFW
dataset, which contains real-world photos with varied pose, lighting, and background rather than
tightly controlled studio images. After normalizing pixel values, reshaping images for the CNN,
and splitting the data 80/20 with stratification to preserve each identity's representation, a CNN
with three convolutional blocks (Conv2D + batch normalization + max pooling + dropout) followed by
a dense classification head was trained to distinguish between individuals. The most influential
factor in performance is the number of training images available per identity — classes with more
example photos are recognized far more reliably than those near the minimum-image cutoff,
reflecting the class-imbalance that's inherent to "in the wild" datasets like LFW. One clear
limitation of this CNN approach is that it treats face recognition as closed-set classification:
it can only recognize identities it was explicitly trained on, and cannot generalize to a
brand-new person without retraining, unlike embedding-based approaches (e.g., FaceNet/Siamese
networks) that learn a general notion of face similarity and can verify or recognize identities
never seen during training.
