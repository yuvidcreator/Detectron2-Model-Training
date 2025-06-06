# For Detectron2

pip install --no-cache-dir -r requirements.txt

```
# On macOS (M1 Chip Only), you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" pip install git+https://github.com/facebookresearch/detectron2.git
```

# Custom Model Training using Detectron2 --

For training a custom instance segmentation model with Detectron2 using 37 classes, the required number of annotated images is not fixed and depends on several factors. Here's a breakdown of considerations:

1. Data Variability:

* **Complexity of Classes:**

  If the 37 classes are visually very distinct, fewer images per class might be needed. If classes are similar, more examples are required to distinguish them.
* **Intra-Class Variation:**

  Each class should have images showcasing variations in pose, lighting, background, and viewpoint.
* **Real-World Scenarios:**

  Include images that reflect the real-world conditions where the model will be used.

2. General Guidelines:

* **Minimum per Class:**

  Aim for at least 50-100 annotated instances per class. For complex classes, hundreds of instances are often better.
* **Total Images:**

  With 37 classes, a minimum of a few thousand annotated images is recommended. A dataset of 5,000-10,000 images is a good starting point for many instance segmentation tasks.
* **Data Augmentation:**

  Techniques such as rotation, scaling, cropping, and color adjustments can artificially expand the dataset.

3. Annotation Quality:

* **Accurate Masks:** Instance segmentation requires precise polygonal masks around each object instance. Inaccurate masks lead to poor model performance.
* **Consistent Annotation:** Maintain consistent annotation guidelines across all images.
* **Label Assist Tools:** Use tools that provide automated label assistance to speed up the annotation process.

4. Detectron2 Specifics:

* **Dataset Registration:** Detectron2 requires the dataset to be registered in a specific format, typically JSON, containing image paths, mask coordinates, and class labels.
* **Training Configuration:** Experiment with different configurations, including learning rates, batch sizes, and data augmentation settings.
* **Pre-trained Models:** Use pre-trained models as a starting point to accelerate training.

5. Iterative Process:

* **Start Small:** Begin with a smaller dataset and evaluate the model's performance.
* **Iterate:** Add more data and refine annotations as needed.
* **Monitor Performance:** Track metrics such as mAP and mask quality.

In summary: While there's no magic number, a dataset of at least 5,000-10,000 annotated images with good variability and precise masks is a reasonable starting point for training a Detectron2 instance segmentation model with 37 classes. Remember that data quality and diversity are more critical than sheer quantity.


# For more --

[https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source)
