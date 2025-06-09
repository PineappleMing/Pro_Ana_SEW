import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
from pandas.plotting import parallel_coordinates
import pandas as pd

# -------------------- 核心分析函数 --------------------
def cluster_based_feature_analysis(pos_features, neg_features, n_clusters=10, top_clusters=3):
    """基于聚类的特征簇分析"""
    # 数据预处理
    scaler = StandardScaler()
    combined = np.vstack([pos_features, neg_features])
    scaler.fit(combined)

    # 降维处理
    pca = PCA(n_components=0.95)
    pos_reduced = pca.fit_transform(scaler.transform(pos_features))
    neg_reduced = pca.transform(scaler.transform(neg_features))

    # 自适应聚类
    def auto_cluster(features):
        best_k, max_score = 2, -1
        for k in range(2, min(n_clusters+2, len(features))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            if score > max_score:
                best_k, max_score = k, score
        return KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(features)

    # 执行聚类
    pos_kmeans = auto_cluster(pos_reduced)
    neg_kmeans = auto_cluster(neg_reduced)

    # 簇评分
    def cluster_score(cluster_data, other_data):
        intra_sim = np.mean(cluster_data @ cluster_data.T)
        inter_sim = np.mean(cluster_data @ other_data.T)
        return (intra_sim - inter_sim) * np.log(len(cluster_data))

    # 选择独特簇
    def get_top_clusters(kmeans, self_data, other_data):
        return sorted([
            (cluster_score(self_data[kmeans.labels_ == i], other_data), i)
            for i in range(kmeans.n_clusters)
        ], reverse=True)[:top_clusters]

    pos_clusters = [i for _,i in get_top_clusters(pos_kmeans, pos_reduced, neg_reduced)]
    neg_clusters = [i for _,i in get_top_clusters(neg_kmeans, neg_reduced, pos_reduced)]

    return (pos_kmeans, pos_clusters), (neg_kmeans, neg_clusters)

def plot_intra_distribution(ax, data, kmeans_model, title, highlight_clusters):
    """组内聚类分布可视化"""
    colors = plt.cm.tab10(np.linspace(0, 1, kmeans_model.n_clusters))
    centers = []
    # 绘制所有簇
    for cluster_id in range(kmeans_model.n_clusters):
        mask = kmeans_model.labels_ == cluster_id
        alpha = 0.7 if cluster_id in highlight_clusters else 0.2
        size = 50 if cluster_id in highlight_clusters else 20
        edgecolor = 'k' if cluster_id in highlight_clusters else None
        centers.append(data[mask].mean(axis=0))
        ax.scatter(data[mask, 0], data[mask, 1],
                   color=colors[cluster_id],
                   alpha=alpha,
                   s=size,
                   edgecolor=edgecolor,
                   linewidth=0.5,
                   label=f'Cluster {cluster_id}')
    centers = np.array(centers)
    ax.scatter(centers[:, 0], centers[:, 1],
               marker='*',
               s=200,
               c='gold',
               edgecolor='k',
               linewidth=1,
               zorder=9,
               label='cluster centers')

    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=6, loc='upper right')

def plot_cross_compare(ax, source_data, compare_data, cluster_model, cluster_ids, title):
    """交叉对比可视化"""
    from scipy.stats import gaussian_kde

    # 绘制对比组背景
    ax.scatter(compare_data[:, 0], compare_data[:, 1],
               c='lightgrey',
               alpha=0.3,
               s=10,
               label='contrast group features')

    # 绘制源组独特簇
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
    for i, cid in enumerate(cluster_ids):
        mask = cluster_model.labels_ == cid
        cluster_points = source_data[mask]

        # 准备数据
        x, y = cluster_points[:, 0], cluster_points[:, 1]
        xy = np.vstack([x, y])

        # 正确创建KDE对象
        kde_obj = gaussian_kde(xy)  # 修正点：保存KDE对象而不是直接调用

        # 生成网格
        xmin, xmax = x.min()-1, x.max()+1
        ymin, ymax = y.min()-1, y.max()+1
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        grid_points = np.vstack([xx.ravel(), yy.ravel()])

        # 计算网格点密度
        z = np.reshape(kde_obj(grid_points), xx.shape)  # 正确调用KDE对象

        # 绘制等高线
        levels = np.linspace(z.min(), z.max(), 6)[1:-1]
        ax.contour(xx, yy, z, levels=levels, colors=[colors[i]], alpha=0.5)

        # 绘制散点
        ax.scatter(x, y,
                   c=[colors[i]],
                   s=30,
                   edgecolor='w',
                   linewidth=0.5,
                   label=f'cluster {cid} (n={len(x)})')

    ax.set_title(f"{title}distribution contrast", fontsize=10)
    ax.legend(fontsize=6)

def plot_parallel_coords(ax, pos_data, neg_data, pos_clusters, neg_clusters, pos_kmeans, neg_kmeans):
    """平行坐标图"""
    # 选择差异最大的10个维度
    diff = np.abs(np.mean(pos_data, axis=0) - np.mean(neg_data, axis=0))
    top_dims = np.argsort(diff)[-10:][::-1]

    # 生成采样数据
    def sample_clusters(data, model, clusters, group_name):
        samples = []
        for c in clusters:
            mask = model.labels_ == c
            cluster_data = data[mask][:, top_dims]
            sample_size = min(50, len(cluster_data))  # 每个簇最多采50个
            samples.append(cluster_data[np.random.choice(len(cluster_data), sample_size)])
        return pd.DataFrame(np.vstack(samples),
                            columns=[f'Dim{i}' for i in range(10)]).assign(Group=group_name)

    # 构建DataFrame
    pos_df = sample_clusters(pos_data, pos_kmeans, pos_clusters, 'Positive')
    neg_df = sample_clusters(neg_data, neg_kmeans, neg_clusters, 'Negative')
    df = pd.concat([pos_df, neg_df])

    # 绘制平行坐标
    parallel_coordinates(df, 'Group', color=['#FF4444', '#4444FF'], ax=ax, alpha=0.7)

    # 美化图形
    ax.set_xticklabels([f'dim {i+1}' for i in top_dims], rotation=45, ha='right')
    ax.set_title("feature dim ）", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def add_confidence_ellipse(ax, data, model, clusters, color):
    """添加置信椭圆"""
    for cid in clusters:
        cluster_data = data[model.labels_ == cid]
        if len(cluster_data) < 10: continue

        # 计算椭圆参数
        cov = np.cov(cluster_data.T)
        lambda_, v = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
        width, height = 3 * np.sqrt(lambda_)

        # 绘制椭圆
        ell = Ellipse(xy=np.mean(cluster_data, axis=0),
                      width=width,
                      height=height,
                      angle=angle,
                      facecolor=color,
                      alpha=0.1)
        ax.add_patch(ell)

# -------------------- 可视化函数 --------------------
def visualize_unique_clusters(pos_features, neg_features, pos_results, neg_results):
    """可视化独特特征簇"""
    (pos_kmeans, pos_clusters) = pos_results
    (neg_kmeans, neg_clusters) = neg_results

    # 创建画布
    plt.figure(figsize=(20, 12))

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    pos_tsne = tsne.fit_transform(pos_features)
    neg_tsne = tsne.fit_transform(neg_features)

    # 子图布局
    ax1 = plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)  # 主视图
    ax2 = plt.subplot2grid((3,4), (0,2))                        # 阳性分布
    ax3 = plt.subplot2grid((3,4), (0,3))                        # 阴性分布
    ax4 = plt.subplot2grid((3,4), (1,2))                        # 交叉对比1
    ax5 = plt.subplot2grid((3,4), (1,3))                        # 交叉对比2
    ax6 = plt.subplot2grid((3,4), (2,0), colspan=4)             # 平行坐标

    # 绘制各子图
    plot_main_view(ax1, pos_tsne, neg_tsne, pos_clusters, neg_clusters, pos_kmeans, neg_kmeans)
    plot_intra_distribution(ax2, pos_tsne, pos_kmeans, 'pos cluster', pos_clusters)
    plot_intra_distribution(ax3, neg_tsne, neg_kmeans, 'neg cluster', neg_clusters)
    plot_cross_compare(ax4, pos_tsne, neg_tsne, pos_kmeans, pos_clusters, 'pos unique')
    plot_cross_compare(ax5, neg_tsne, pos_tsne, neg_kmeans, neg_clusters, 'neg unique')
    plot_parallel_coords(ax6, pos_features, neg_features, pos_clusters, neg_clusters, pos_kmeans, neg_kmeans)

    plt.tight_layout()
    plt.show()

def plot_main_view(ax, pos_data, neg_data, pos_clusters, neg_clusters, pos_model, neg_model):
    """主分布视图"""
    # 绘制背景点
    ax.scatter(neg_data[:,0], neg_data[:,1], c='lightgrey', alpha=0.3, s=10)
    ax.scatter(pos_data[:,0], pos_data[:,1], c='silver', alpha=0.3, s=10)

    # 绘制独特簇
    colors = ['red', 'darkorange', 'gold']
    for i, cid in enumerate(pos_clusters):
        mask = pos_model.labels_ == cid
        ax.scatter(pos_data[mask,0], pos_data[mask,1],
                  c=colors[i], s=30, edgecolor='k', label=f'pos_{cid}')
    for i, cid in enumerate(neg_clusters):
        mask = neg_model.labels_ == cid
        ax.scatter(neg_data[mask,0], neg_data[mask,1],
                  c=colors[i], s=30, marker='s', edgecolor='k', label=f'neg_{cid}')

    # 添加图例和标题
    ax.legend()
    ax.set_title("feature distribution (t-SNE)", fontsize=12)

# 其他可视化函数实现（完整代码需包含所有子函数）
# ...（由于篇幅限制，此处需包含前文提到的所有子函数实现）...

# -------------------- 示例使用 --------------------
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    pos_data = np.vstack(emb_list_1)
    neg_data = np.vstack(emb_list_0)


    # 执行聚类分析
    pos_res, neg_res = cluster_based_feature_analysis(pos_data, neg_data)

    # 可视化结果
    visualize_unique_clusters(pos_data, neg_data, pos_res, neg_res)