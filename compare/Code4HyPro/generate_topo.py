import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# 可选：同时限制其他库的线程数以防冲突
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import pickle
import argparse
from collections import defaultdict

def generate_topo_protos(seq_file_path, output_path, n_clusters=500, filter_threshold=1, embedding_dim=128):
    """
    根据 all_train_seq.txt 生成拓扑原型文件
    
    Args:
        seq_file_path: all_train_seq.txt 的路径 (pickle格式的 list of lists)
        output_path: 输出 .pkl 文件的路径
        n_clusters: 聚类数量 (对应论文中的 k_t, 默认为 500)
        filter_threshold: 过滤边的阈值 (默认为 3)
        embedding_dim: 谱聚类的嵌入维度 (对应论文中的 q, 推荐 128)
    """
    
    # ---------------------------------------------------------
    # 1. 加载数据
    # ---------------------------------------------------------
    print(f"Loading sequences from {seq_file_path}...")
    try:
        with open(seq_file_path, 'rb') as f:
            all_seqs = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # ---------------------------------------------------------
    # 2. 构建全局图 (Global Graph Construction)
    # ---------------------------------------------------------
    print("Building global graph...")
    # 使用字典统计边权重: (u, v) -> count
    edge_counts = defaultdict(int)
    nodes = set()
    
    for seq in all_seqs:
        # 确保 seq 是列表
        if not isinstance(seq, (list, np.ndarray)):
            continue
        
        # 记录节点并统计相邻边
        for i in range(len(seq) - 1):
            u = seq[i]
            v = seq[i+1]
            nodes.add(u)
            nodes.add(v)
            
            # 无向图，统一存储为 (min, max)
            if u > v:
                u, v = v, u
            edge_counts[(u, v)] += 1
            
    # 处理最后一个节点（如果它没在前序对中出现过，虽然不太可能，但为了保险）
    if len(all_seqs) > 0:
        for seq in all_seqs:
            if len(seq) > 0:
                nodes.add(seq[-1])

    # 映射 Item ID 到 矩阵索引 (0 ~ N-1)
    # 因为原始 Item ID 可能很大且不连续，需要重新映射以构建紧凑矩阵
    sorted_nodes = sorted(list(nodes))
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
    idx_to_node = {i: node for i, node in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)
    max_item_id = max(sorted_nodes) if sorted_nodes else 0
    
    print(f"Found {num_nodes} unique items. Max Item ID: {max_item_id}")

    # ---------------------------------------------------------
    # 3. 过滤与构建矩阵 (Filtering & Matrix Construction)
    # ---------------------------------------------------------
    print(f"Filtering edges with weight < {filter_threshold}...")
    row = []
    col = []
    data = []
    
    filtered_edges_count = 0
    for (u, v), count in edge_counts.items():
        if count >= filter_threshold:
            i = node_to_idx[u]
            j = node_to_idx[v]
            
            # 构建对称矩阵 (无向图)
            row.extend([i, j])
            col.extend([j, i])
            data.extend([count, count]) # 权重为共现次数
            filtered_edges_count += 1
            
    print(f"Kept {filtered_edges_count} edges after filtering.")
    
    # 构建稀疏邻接矩阵 W
    W = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # ---------------------------------------------------------
    # 4. 计算拉普拉斯矩阵 (Laplacian Calculation)
    # ---------------------------------------------------------
    print("Calculating Normalized Laplacian...")
    # 计算度矩阵 D
    degrees = np.array(W.sum(axis=1)).flatten()
    # D^(-1/2)
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees!=0)
    d_inv_sqrt[degrees == 0] = 0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # L = I - D^(-1/2) * W * D^(-1/2)
    I = sp.eye(num_nodes)
    L = I - D_inv_sqrt @ W @ D_inv_sqrt
    
    # ---------------------------------------------------------
    # 5. 特征分解 (Eigen Decomposition)
    # ---------------------------------------------------------
    print(f"Computing top-{embedding_dim} eigenvectors...")
    # 计算最小的 q 个特征值对应的特征向量
    # which='SM' (Smallest Magnitude)
    # 注意：特征值可能包含接近0的值，这是连通分量导致的，是正常的
    try:
        # eigsh 需要 k < N
        k_eig = min(embedding_dim, num_nodes - 1)
        vals, vecs = eigsh(L, k=k_eig, which='SM')
    except Exception as e:
        print(f"Eigen decomposition failed (graph might be too small?): {e}")
        # fallback for very small graphs in testing
        vals, vecs = np.linalg.eigh(L.toarray())
        vecs = vecs[:, :embedding_dim]

    # vecs 是 (num_nodes, embedding_dim) 的矩阵，即论文中的 V
    
    # ---------------------------------------------------------
    # 6. K-Means 聚类 (Clustering)
    # ---------------------------------------------------------
    print(f"Running K-Means with k={n_clusters}...")
    # 确保聚类数不超过样本数
    actual_k = min(n_clusters, num_nodes)
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vecs)
    
    # ---------------------------------------------------------
    # 7. 保存结果 (Mapping & Saving)
    # ---------------------------------------------------------
    print("Saving results...")
    
    # main.py 中通过 pickle.load 加载后直接用 torch.tensor 转换
    # 这意味着它期望一个列表或数组，索引对应 ItemID
    # 所以我们需要创建一个大小为 max_item_id + 1 的数组
    
    # 初始化全为 0 (或者一个默认类别)
    # 注意：如果某个 ItemID 在训练集中不存在（比如只在测试集出现，或者被过滤掉了），
    # 这里默认给它分配类别 0 或者其他逻辑
    final_protos = np.zeros(max_item_id + 1, dtype=int)
    
    for i, label in enumerate(labels):
        original_item_id = idx_to_node[i]
        final_protos[original_item_id] = label
        
    # 保存为 pkl
    with open(output_path, 'wb') as f:
        pickle.dump(final_protos, f) # 保存为 numpy 数组或 list
        
    print(f"Successfully saved to {output_path}")
    print(f"Output shape: {final_protos.shape}")
    print(f"Example (Item 1 -> Proto): {final_protos[1] if len(final_protos) > 1 else 'N/A'}")

# ==========================================
# 测试用例 (Test Case)
# ==========================================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/xuguangning/work/UIO-LLM-SBR/datasets/', help='Path to all_train_seq.txt')
    parser.add_argument('--dataset', type=str, default='diginetica', help='Path to all_train_seq.txt')

    args, unknown = parser.parse_known_args()
    input_path = os.path.join(args.path, args.dataset, 'all_train_seq.txt')
    output_path = os.path.join(args.path, args.dataset, 'topo_protos_ratio_500.pkl')
    
    if input_path and output_path:
        generate_topo_protos(input_path, output_path)