import numpy as np
import umap
import json
import random
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)
random.seed(42)

# 参数配置
num_samples = 300              # 总样本数
embedding_dim = 512            # embedding 维度
num_clusters = 8               # cluster 数量（即组织区域）
num_labels = 2                 # label 数量：阴性和阳性

# 每个 cluster 对应的 label 分布倾向（越接近 1 表示阳性越多）
cluster_label_bias = {
    i: random.uniform(0.3, 0.7) for i in range(num_clusters)
}  # 控制每个 cluster 中 label 的分布比例，让绝大多数 cluster 同时包含两种 label

# 生成 cluster 中心
centers = np.random.uniform(-5, 5, size=(num_clusters, embedding_dim))

# 存储数据
patch_coords = []
slide_ids = []
embeddings = []
labels = []
image_urls = []

for i in range(num_samples):
    assigned_cluster = random.randint(0, num_clusters - 1)
    
    # 按照 cluster 的 label 偏好决定当前样本是阴性还是阳性
    positive_prob = cluster_label_bias[assigned_cluster]
    label = "1" if random.random() < positive_prob else "0"
    
    # 生成 embedding（围绕该 cluster 中心的小扰动）
    embedding = np.random.normal(loc=centers[assigned_cluster], scale=0.8)
    
    # 添加数据项
    patch_coords.append([random.randint(0, 1000), random.randint(0, 1000)])
    slide_ids.append(f"slide_{random.randint(1, 5)}")
    embeddings.append(embedding)
    labels.append(label)
    random_int = random.randint(1, 6)
    label_type = "pos" if label == "1" else "neg"
    image_urls.append(f"/mock_images/{label_type}_{random_int}.png")


# 转换为 numpy 数组
embeddings = np.array(embeddings)

# 使用 UMAP 降维
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings).tolist()
umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(embeddings).tolist()

# 构建 JSON 数据
data = []
for i in range(num_samples):
    data.append({
        "patch_coord": patch_coords[i],
        "slide_id": slide_ids[i],
        "embedding": embeddings[i].tolist(),
        "label": labels[i],
        "image_url": image_urls[i],
        "UMAP_2d": umap_2d[i],
        "UMAP_3d": umap_3d[i]
    })

# 保存文件
with open("./public/clustered_patch_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 已生成多 cluster 数据，并保存为 clustered_patch_data.json")