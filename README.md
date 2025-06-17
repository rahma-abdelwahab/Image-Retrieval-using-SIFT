# 🖼️ Image Retrieval System using SIFT

This project implements a content-based image retrieval system that uses **SIFT (Scale-Invariant Feature Transform)** for feature extraction and  **TF-IDF** weighting for image representation. The system supports image similarity search using **cosine similarity**, enabling retrieval of visually similar images from a dataset.

---

## 🔍 Features

* **🔑 SIFT Feature Extraction**
  Extracts robust and scale-invariant keypoints and descriptors from each image using OpenCV's SIFT implementation.

* **📦 Visual Vocabulary Creation**
  Aggregates SIFT descriptors and applies **K-Means clustering** to form a visual vocabulary (codebook) of "visual words."

* **📚 TF-IDF Weighting**
  Enhances BoW vectors by applying **Term Frequency–Inverse Document Frequency (TF-IDF)** to give higher weight to discriminative visual words.

* **📈 Cosine Similarity Matching**
  Computes cosine similarity between query and dataset vectors to rank and retrieve relevant images.

* **⚙️ Parameter Optimization**
  Includes experimentation for:

  * Optimal number of K-Means centroids
  * Best subset of images for codebook creation
  * Performance comparison between BoW and TF-IDF vectors

---

## 🛠️ Technologies Used

| Category                | Libraries / Tools                            |
| ----------------------- | -------------------------------------------- |
| Programming Language    | Python                                       |
| Feature Extraction      | OpenCV (`cv2`)                               |
| Clustering & Similarity | Scikit-learn (`KMeans`, `cosine_similarity`) |
| Array Ops & Math        | NumPy (`numpy`)                              |
| Visualization           | Matplotlib (`pyplot`)                        |
| Progress Tracking       | TQDM                                         |
| File Management         | OS (`os`)                                    |

---

> ⚠️ Make sure to update the `data_path` in the notebook to point to your dataset folder.

---

## ▶️ How to Run

1. **Install Dependencies**:

   ```bash
   pip install opencv-python numpy scikit-learn matplotlib tqdm
   ```

2. **Prepare Image Dataset**:

   * Place your image files inside a directory, e.g., `dataset/`.
   * Update the `data_path` variable inside the notebook accordingly:

     ```python
     data_path = '/path/to/your/dataset/'
     ```

3. **Run the Notebook**:

   * Open `Image_Retrieval_using_SIFT.ipynb` using Jupyter Lab or Google Colab.
   * Run all cells sequentially.
   * The notebook will perform:

     * Feature extraction
     * Vector representation
     * Retrieval visualization and accuracy analysis

---

## 📌 Use Cases

* Visual search engines
* Duplicate or similar image detection
* Content-based image indexing for digital libraries
* Educational demos for  SIFT applications in computer vision

