# ======================================================================
#  HIGH-ACCURACY GMM SPEAKER RECOGNITION 
#  - Features: 20 MFCCs (High Resolution)
#  - Hop Length: 10ms (3x Data Density)
#  - Pre-emphasis: Enabled
#  - VAD & CMVN: Enabled
# ======================================================================

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import hashlib
import random
import time
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from scipy.signal import wiener, lfilter
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, pairwise
from collections import defaultdict

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
# Update this path to your local dataset
DATASET_PATH = r"C:\Users\pavankalyan\Desktop\vox1_dev_wav\wav"

# We use a specific high-acc cache to keep things clean
CACHE_DIR = "mfcc_cache_high_acc" 
CHECKPOINT_FILE = "results_checkpoint_high_acc.pkl"

# We push the components higher because we have more data now
GMM_COMPONENTS_LIST = [32, 64, 128, 256, 512] 
MFCC_DIMS_LIST = [20] 
IVECTOR_DIMS_LIST = [20, 60] 

# Use maximum CPU cores
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def _ivector_default_factory(): return defaultdict(dict)

def _cache_key_for_path(path, n_dims):
    return hashlib.md5((os.path.abspath(path) + f"_{n_dims}_hires").encode()).hexdigest()

# ---------------------------------------------------
# 1) Optimized Feature Extraction
# ---------------------------------------------------
def extract_and_cache_mfcc(file_path, n_dims=20):
    try:
        key = _cache_key_for_path(file_path, n_dims)
        cache_file = os.path.join(CACHE_DIR, f"{key}.npy")
        if os.path.exists(cache_file):
            return np.load(cache_file, allow_pickle=False)

        y, sr = librosa.load(file_path, sr=16000)
        if y is None or len(y) < 400: return np.array([])

        # 1. Pre-emphasis
        y = lfilter([1, -0.97], [1], y)

        # 2. Noise Reduction
        try:
            y_nr = nr.reduce_noise(y=y, sr=sr, n_fft=512, hop_length=160)
        except:
            y_nr = y

        # 3. VAD (Remove Silence)
        intervals = librosa.effects.split(y_nr, top_db=25, frame_length=512, hop_length=160)
        if len(intervals) == 0: return np.array([])
        y_vad = np.concatenate([y_nr[start:end] for start, end in intervals])
        
        if len(y_vad) < 800: return np.array([])

        # 4. MFCC Extraction (High Resolution: 10ms hop)
        mfccs = librosa.feature.mfcc(y=y_vad, sr=sr, n_mfcc=20, n_fft=400, hop_length=160)
        
        # Deltas
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        combined = np.concatenate((mfccs, delta, delta2)) 
        out = combined.T

        # 5. CMVN
        scaler = StandardScaler()
        out = scaler.fit_transform(out)

        np.save(cache_file, out)
        return out
    except Exception:
        return np.array([])

# ---------------------------------------------------
# 2) Dataset Loading
# ---------------------------------------------------
def load_dataset(dataset_path, train_ratio=0.8, n_dims=20, max_speakers=100):
    train_data, test_data = {}, {}
    speaker_count = 0
    print(f"[Data] Loading High-Res Dataset (max_speakers={max_speakers})...")

    for speaker in sorted(os.listdir(dataset_path)):
        if max_speakers is not None and speaker_count >= max_speakers: break
        spk_path = os.path.join(dataset_path, speaker)
        if not os.path.isdir(spk_path): continue

        files = []
        for root, _, filenames in os.walk(spk_path):
            files.extend([os.path.join(root, f) for f in filenames if f.lower().endswith(".wav")])

        if len(files) < 10: continue

        random.shuffle(files)
        split_idx = max(1, int(train_ratio * len(files)))
        train_files, test_files = files[:split_idx], files[split_idx:]

        feats = []
        for f in train_files:
            mfcc = extract_and_cache_mfcc(f, n_dims)
            if mfcc.size > 0: feats.append(mfcc)
        
        if not feats: continue

        train_data[speaker] = np.vstack(feats)
        test_data[speaker] = test_files
        speaker_count += 1

    print(f"  -> Loaded {len(train_data)} speakers.")
    return train_data, test_data

# ---------------------------------------------------
# 3) Training & Evaluation
# ---------------------------------------------------
def _train_one_gmm(item, n_components):
    spk, feats = item
    if len(feats) < n_components: return spk, None
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag',
            max_iter=200,
            random_state=42,
            n_init=1,
            verbose=0,
            reg_covar=1e-4
        )
        gmm.fit(feats)
        return spk, gmm
    except: return spk, None

def train_speaker_gmms(data, n_components, n_jobs=N_JOBS):
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_train_one_gmm)(itm, n_components) for itm in list(data.items()))
    return {spk: gmm for spk, gmm in results if gmm is not None}

def get_supervectors(models):
    return {spk: gmm.means_.flatten() for spk, gmm in models.items()}

def compute_ivectors(supervectors, desired_dim=60):
    speakers = list(supervectors.keys())
    if not speakers: return {}, None, None
    X = np.vstack([supervectors[s] for s in speakers])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    n_components = min(desired_dim, len(speakers), X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=0).fit(X_scaled)
    ivectors_array = pca.transform(X_scaled)
    return {spk: ivectors_array[i] for i, spk in enumerate(speakers)}, scaler, pca

def _score_file_with_models(args):
    spk, filepath, models, n_dims = args
    feats = extract_and_cache_mfcc(filepath, n_dims)
    if feats.size == 0 or not models: return None
    try:
        scores = {m_spk: gmm.score(feats) for m_spk, gmm in models.items()}
        return (spk, max(scores, key=scores.get))
    except: return None

def evaluate_gmm(models, test_data, n_dims, n_jobs=N_JOBS):
    tasks = [(spk, f, models, n_dims) for spk, files in test_data.items() for f in files]
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_score_file_with_models)(t) for t in tasks)
    results = [r for r in results if r]
    return accuracy_score([r[0] for r in results], [r[1] for r in results]) if results else 0.0

def evaluate_vectors(enrolled_vectors, test_data, n_dims, n_components, scaler=None, pca=None):
    y_true, y_pred = [], []
    for spk, test_files in test_data.items():
        for f in test_files:
            try:
                test_feats = extract_and_cache_mfcc(f, n_dims)
                if test_feats.size == 0: continue
                gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0, n_init=1, reg_covar=1e-4)
                gmm.fit(test_feats)
                test_vec = gmm.means_.flatten().reshape(1, -1)
                
                if scaler and pca: test_vec = pca.transform(scaler.transform(test_vec))
                
                scores = {e_spk: pairwise.cosine_similarity(test_vec, e_vec.reshape(1, -1))[0][0] for e_spk, e_vec in enrolled_vectors.items()}
                y_true.append(spk)
                y_pred.append(max(scores, key=scores.get))
            except: continue
    return accuracy_score(y_true, y_pred) if y_true else 0.0

def save_results_to_file(results, filename="results.txt"):
    with open(filename, "w") as f:
        f.write("="*80 + "\n  HIGH ACCURACY SPEAKER RECOGNITION RESULTS\n" + "="*80 + "\n\n")
        for method in ['GMM', 'Supervector']:
            f.write(f"\n--- {method} Accuracy ---\n" + "-"*60 + "\n")
            f.write("Features            " + "  ".join([f"GMM-{c:<5}" for c in GMM_COMPONENTS_LIST]) + "\n")
            for mfcc_dim in sorted(results[method].keys()):
                accs = [results[method][mfcc_dim].get(c, 0.0) for c in GMM_COMPONENTS_LIST]
                line = f"20 MFCC + Deltas    "
                for val in accs: line += f"{val*100:6.2f}%  "
                f.write(line + "\n")
        f.write("\n" + "="*80 + "\n")

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    print("="*80 + "\nSTARTING HIGH-ACCURACY EXPERIMENT (100 Speakers)\n" + "="*80)
    
    # --- SAFE CHECKPOINT LOADING ---
    if os.path.exists(CHECKPOINT_FILE):
        try: 
            results = pd.read_pickle(CHECKPOINT_FILE)
            print("✅ Checkpoint loaded.")
        except: 
            # If load fails, start fresh
            results = {'GMM': defaultdict(dict), 'Supervector': defaultdict(dict), 'i-vector': {}}
    else: 
        results = {'GMM': defaultdict(dict), 'Supervector': defaultdict(dict), 'i-vector': {}}

    MAX_SPEAKERS = 1000 

    for n_dims in MFCC_DIMS_LIST:
        train_data, test_data = load_dataset(DATASET_PATH, n_dims=n_dims, max_speakers=MAX_SPEAKERS)
        if len(train_data) < 2: continue

        for n_components in GMM_COMPONENTS_LIST:
            print(f"\n[RUNNING] MFCC: 20 (+Deltas=60D) | GMM: {n_components} Comp")
            
            # 1. GMM
            gmm_models = train_speaker_gmms(train_data, n_components, n_jobs=N_JOBS)
            if len(gmm_models) < 2: continue
            
            acc = evaluate_gmm(gmm_models, test_data, n_dims, n_jobs=N_JOBS)
            results['GMM'][n_dims][n_components] = acc
            print(f"  👉 GMM Acc: {acc*100:.2f}%")
            
            # 2. Supervector
            super_vecs = get_supervectors(gmm_models)
            acc_sup = evaluate_vectors(super_vecs, test_data, n_dims, n_components)
            results['Supervector'][n_dims][n_components] = acc_sup
            print(f"  👉 Supervec Acc: {acc_sup*100:.2f}%")
            
            # 3. i-vector (SAFE SAVE)
            for iv_dim in IVECTOR_DIMS_LIST:
                ivecs, scaler, pca = compute_ivectors(super_vecs, desired_dim=iv_dim)
                if ivecs:
                    acc_iv = evaluate_vectors(ivecs, test_data, n_dims, n_components, scaler, pca)
                    
                    # --- SAFE DICTIONARY SAVING (PREVENTS CRASH) ---
                    if 'i-vector' not in results: results['i-vector'] = {}
                    if n_dims not in results['i-vector']: results['i-vector'][n_dims] = {}
                    if iv_dim not in results['i-vector'][n_dims]: results['i-vector'][n_dims][iv_dim] = {}
                    
                    results['i-vector'][n_dims][iv_dim][n_components] = acc_iv
                    print(f"  👉 i-vec ({iv_dim}D) Acc: {acc_iv*100:.2f}%")
            
            pd.to_pickle(results, CHECKPOINT_FILE)
            print("  💾 Saved.")

    save_results_to_file(results)
    print("\n=== DONE ===")