🎙️ Text-Independent Speaker Recognition System (High-Accuracy GMM)

This repository contains a high-accuracy, text-independent speaker recognition system built using classical statistical modeling techniques. The system is designed to handle large-scale audio data and focuses on robust feature engineering, overfitting mitigation, and efficient training on the VoxCeleb dataset.

The project was developed as part of a UROP research initiative, emphasizing signal processing fundamentals and scalable ML system design rather than deep learning.

🚀 Key Highlights

Text-independent speaker identification using Gaussian Mixture Models (GMMs)

High-resolution MFCC feature extraction with Delta and Delta-Delta coefficients

Voice Activity Detection (VAD) and Cepstral Mean Variance Normalization (CMVN) to reduce overfitting

Scalable caching system for large-scale audio processing

Supports GMM, Supervector, and i-vector based evaluation

Optimized for multi-core CPU execution

📊 Dataset

Dataset: VoxCeleb1 (30GB+ audio corpus)

Sampling Rate: 16 kHz

Max Speakers (configurable): Up to 1000

Task: Closed-set speaker identification

⚠️ Dataset is not included due to licensing constraints.
Download from the official VoxCeleb website.

🧠 System Architecture
1. Feature Extraction Pipeline

Pre-emphasis filtering

Noise reduction

Voice Activity Detection (silence removal)

MFCC extraction:

20 MFCC coefficients

Delta and Delta-Delta derivatives (60D total)

10 ms hop length (3× temporal resolution)

Cepstral Mean Variance Normalization (CMVN)

All extracted features are cached on disk to avoid recomputation.

2. Modeling Approaches

The system evaluates three speaker modeling techniques:

🔹 Gaussian Mixture Models (GMM)

Diagonal covariance

Components tested: 32, 64, 128, 256, 512

Speaker identification via maximum log-likelihood scoring

🔹 Supervectors

GMM mean vectors flattened into fixed-length representations

Similarity scoring using cosine similarity

🔹 i-vectors

PCA-based dimensionality reduction on supervectors

Dimensions tested: 20, 60

Cosine similarity-based classification

⚙️ Configuration

Key parameters can be modified at the top of the script:

DATASET_PATH = "path/to/vox1_dev_wav"
GMM_COMPONENTS_LIST = [32, 64, 128, 256, 512]
MFCC_DIMS_LIST = [20]
IVECTOR_DIMS_LIST = [20, 60]
MAX_SPEAKERS = 1000

🧪 Training & Evaluation

Speaker-wise train/test split (80/20)

Parallelized training using joblib

Accuracy computed using closed-set identification

Checkpointing enabled to safely resume long experiments

Sample output:

GMM Accuracy:        86.70%
Supervector Accuracy: 2.16%
i-vector Accuracy:   3.00%

💾 Performance Optimizations

Disk-based MFCC caching reduces repeated feature extraction

Checkpointing prevents loss of progress during long experiments

CPU thread limiting ensures stable execution on shared environments

Designed to run efficiently on Google Colab and local machines

🛠️ Tech Stack

Language: Python

Machine Learning: Scikit-learn

Audio Processing: Librosa, SciPy, noisereduce

Numerical Computing: NumPy

Parallelization: Joblib

Dimensionality Reduction: PCA

Environment: Google Colab / Local CPU

📁 Repository Structure
.
├── speaker_recognition.py
├── mfcc_cache_high_acc/
├── results_checkpoint_high_acc.pkl
├── results.txt
└── README.md

🔧 How to Run
Clone the repository:

https://github.com/ramakrishnakola507/Text_independent_speaker_recognition
cd Text_independent_speaker_recognition

2. Install Dependencies

It is recommended to use a virtual environment.

pip install -r requirements.txt

3. Configure Dataset Path

Update the following variable in main.py to point to your local VoxCeleb1 dataset:

DATASET_PATH = "path/to/vox1_dev_wav"

4. Run the Pipeline
python main.py

⚠️ Note: This experiment is computationally intensive. For large speaker counts, running on Google Colab or a multi-core CPU machine is recommended.

📌 Results Summary
Method	Best Accuracy
GMM	86.70%
Supervector	~2.16%
i-vector	~3.00%

Accuracy improved significantly after integrating VAD + CMVN, addressing severe overfitting observed in baseline models.

🔍 Key Learnings

Classical ML models can remain competitive with strong feature engineering

Overfitting in speaker recognition is heavily influenced by silence and channel effects

Efficient data handling is critical when working with large audio datasets

System-level optimizations matter as much as model selection

📜 License & Disclaimer

This project is intended for educational and research purposes only.
VoxCeleb data is subject to its original licensing terms.
