import os
import numpy as np
from sklearn.model_selection import KFold

data_dir = "ROOT_DIR"
save_dir = './demo/TCGA-PAAD/splits'
os.makedirs(save_dir, exist_ok=True)
all_files = sorted(os.listdir(data_dir))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, val_idx) in enumerate(kf.split(all_files)):
    train_files = [all_files[j] for j in train_idx]
    val_files = [all_files[j] for j in val_idx]
    folders = [train_files, val_files]

    save_path = os.path.join(save_dir, f'folder_{i+1}.npy')
    np.save(save_path, folders)