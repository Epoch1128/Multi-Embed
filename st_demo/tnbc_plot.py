import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
import os

def visual_tnbc(feature, coords, mask, save_dir, with_spatial=True, cluster_num=6, random_seed=42):
    os.makedirs(save_dir, exist_ok=True)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature)
    coords_scaled = scaler.fit_transform(coords)
    if with_spatial:
        combined_data = np.hstack((features_scaled, coords_scaled))
    else:
        combined_data = features_scaled
    kmeans = KMeans(n_clusters=cluster_num, random_state=random_seed)
    kmeans_labels = kmeans.fit_predict(combined_data)

    height, width = mask.shape

    rgb_list = [
        (51, 87, 255),
        (238, 68, 68),
        (255, 204, 0),
        (34, 170, 170),
        (168, 216, 185),
        (200, 0, 0),
        (0, 255, 0)
    ]
    image = np.ones((284, 284, 3), dtype=np.uint8) * 255
    for x, y, label in zip(coords[:, 0], coords[:, 1], kmeans_labels):
        resize_x, resize_y = x // 112, y // 112
        image[resize_y, resize_x, :] = np.array(rgb_list[label], dtype=np.uint8)
    # plt.imshow(image)
    image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LINEAR)
    image[mask == 0] = (255, 255, 255)
    cv2.imwrite(os.path.join(save_dir, 'cluster.png'), image)
    print(f"Visualization results have been saved in {save_dir}")

# res_pth = '/results/save/st/CN15_D2/CN15_D2_sr.pkl'
# mask_pth = '/code/tutorials/CN15_d2_mask.png'
# save_dir = '/results/save/st/CN15_D2/'
# visual_tnbc(res_pth, mask_pth, save_dir)