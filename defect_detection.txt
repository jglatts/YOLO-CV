# Defect Detection with YOLO

## Why YOLO works for defect detection

### Object detection framework
YOLO detects objects in images by drawing bounding boxes and assigning class labels. A defect can be treated as an “object” in this context.

### Real-time performance
YOLO is fast, so you can implement **live quality inspection** on production lines with USB or industrial cameras.

### Flexible for small datasets
YOLOv8 (or YOLOv11) works well even with limited data if you use **augmentation** or **transfer learning**.

---

## How to set up defect detection

### 1. Collect defect images
- Gather images of your parts with defects (e.g., scratches, misalignments, missing components, contamination).  
- Also include **good (non-defective) parts** as negative samples.

### 2. Label your images
- Use labeling tools like [LabelImg](https://github.com/heartexlabs/labelImg) or [Roboflow](https://roboflow.com/).  
- Draw bounding boxes around defects and assign classes (e.g., `scratch`, `crack`, `misaligned`).

### 3. Create a dataset YAML
Example `dataset.yaml`:

```yaml
train: ./datasets/train/images
val: ./datasets/val/images

nc: 3
names: ['scratch', 'crack', 'misaligned']
