import pickle
import os
import argparse
import numpy as np

def generate_global_graph_data(dataset_path, output_filename='all_train_seq.txt'):
    """
    读取 train.txt (格式为 [inputs_list, targets_list])
    将 input 和 target 拼接，生成用于构建全局图的序列文件
    """
    train_file = os.path.join(dataset_path, 'train.txt')
    
    if not os.path.exists(train_file):
        print(f"Error: 找不到文件 {train_file}")
        return

    print(f"正在加载 {train_file} ...")
    # train_data 通常是一个元组: (sequences, targets)
    # sequences 是一个列表的列表 [[1,2], [1,2,5], ...]
    # targets 是一个列表 [3, 6, ...]
    train_data = pickle.load(open(train_file, 'rb'))
    
    inputs = train_data[0]
    targets = train_data[1]
    
    all_seqs = []
    
    print(f"正在处理 {len(inputs)} 条数据...")
    
    for seq, target in zip(inputs, targets):
        # 确保 seq 是列表格式
        if isinstance(seq, np.ndarray):
            seq = seq.tolist()
        elif not isinstance(seq, list):
            seq = list(seq)
            
        # 关键步骤：拼接 Input 和 Target
        # 因为在全局图中，Sequence最后一个点 -> Target 这个跳转是非常重要的共现关系
        full_seq = seq + [target]
        all_seqs.append(full_seq)

    # 结果是一个 list of lists
    output_path = os.path.join(dataset_path, output_filename)
    
    print(f"正在保存到 {output_path} ...")
    with open(output_path, 'wb') as f:
        pickle.dump(all_seqs, f)
        
    print("完成！")
    print(f"示例数据 (前3条): {all_seqs[:3]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 修改这里 default 为你的数据集文件夹路径
    parser.add_argument('--dataset_dir', default='/data/UIO-SBR/datasets/Tmall', 
                        help='包含 train.txt 的文件夹路径')
    opt = parser.parse_args()

    generate_global_graph_data(opt.dataset_dir)