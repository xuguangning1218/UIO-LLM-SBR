import random
import json
import pdb
import numpy as np
from random import shuffle,randint,choice,sample
import argparse
import os
import pickle


def identify_sbr_users(x_list, y_list):
    """
    将SBR的x和y列表重组为以user为单位的字典。
    逻辑：如果 x[i] == x[i+1] + [y[i+1]]，则视为同一用户。
    """
    grouped_data = {}
    user_count = 0
    
    # 暂存当前正在处理的 user 数据
    current_user_data = []
    
    # 遍历数据 (一直到倒数第二个，因为要和 i+1 比较)
    for i in range(len(x_list)):
        # 将当前行加入暂存区
        current_user_data.append([x_list[i], y_list[i]])
        
        # 边界检查：如果是最后一行，或者不满足连续性条件，则结算当前用户
        is_last_item = (i == len(x_list) - 1)
        
        if not is_last_item:
            current_x = x_list[i]
            next_x = x_list[i+1]
            next_y = y_list[i+1]
            
            # 核心判断逻辑：判断是否为包含关系（倒序生成）
            # 注意：这里假设 list 元素是 int，可以直接 + 拼接
            if current_x != (next_x + [next_y]):
                # 这是一个新用户的断点
                grouped_data[f"{user_count}"] = current_user_data
                user_count += 1
                if user_count < 10: print(current_user_data)
                current_user_data = [] # 重置
        else:
            # 循环结束，保存最后积累的数据
            grouped_data[f"{user_count}"] = current_user_data

    return grouped_data


def generate_sbr_jsonl(result_dict, user_id_aug_size, output_file="sbr_augmented.jsonl", keep_prob=1.0):
    """
    将 result_dict 转换为 JSONL 格式的指令微调数据。
    
    逻辑：
    1. 提取每个 User 的所有交互 Item (Set去重)。
    2. 严格执行 Code A 的逻辑：随机打乱 (Shuffle) -> 随机切分 -> 生成 20 条样本。
    3. 写入 .jsonl 文件 (每行一个 json 对象)。
    """
    
    # 使用 'w' 模式打开文件，准备写入
    with open(output_file, 'w', encoding='utf-8') as f_output:
        
        # 遍历 result_dict 中的每个用户
        for user_id, sessions in result_dict.items():

            # --- [核心修改] 用户采样逻辑 ---
            # 如果随机数大于保留概率，则跳过该用户 (实现降采样)
            if random.random() > keep_prob:
                continue
            
            # --- 步骤 1: 数据扁平化 & 去重 (Flatten) ---
            # 我们不关心时序，只关心这个用户交互过哪些物品的集合
            unique_items = set()
            for x_seq, y_val in sessions:
                # x_seq 是一个列表，update 可以批量添加
                unique_items.update(x_seq)
                # y_val 是一个数值，add 添加单个元素
                unique_items.add(y_val)
            
            # 转为列表，这就是 Code A 中的 positive_list
            positive_list = list(unique_items)
            len_positive_list = len(positive_list)
            
            # --- 步骤 2: 严格复刻 Code A 增强逻辑 ---
            
            # 逻辑: 数据过少则跳过 (至少需要3个物品才能切分 input 和 label)
            if len_positive_list < 3:
                continue
                
            # 逻辑: 确定采样次数 (默认20次)
            # 保护措施：如果总长度不够减2，就无法进行randint，需要跳过
            if len_positive_list - 2 < 1:
                continue
                
            # 生成 20 个随机切分点 (Code A 逻辑)
            # size=20 意味着每个用户生成 20 条增强数据
            rand_sel = np.random.randint(len_positive_list - 2, size=user_id_aug_size)
            
            for item_j in rand_sel:
                # 切分点索引
                item_i = item_j + 1
                
                # [核心逻辑]: 每次循环都重新打乱顺序 (Shuffle)
                # 这保证了模型学习的是“物品集合”而非“固定序列”
                random.shuffle(positive_list)
                
                # --- 构造 Input (q) ---
                # 获取列表的前半部分
                input_items = positive_list[:item_i]
                # 转换为字符串，添加 'i' 前缀 (如: i10,i12,i5)
                preference_str = ",".join([f"i{item}" for item in input_items])
                
                # --- 构造 Output (a) ---
                # 获取列表的后半部分
                target_items = positive_list[item_i:]
                # 转换为字符串
                pos_str = ",".join([f"i{item}" for item in target_items])
                
                # --- 构造 JSON 对象 ---
                str_out = {
                    "prompt": f"Given the user(u{user_id})'s clicked list items:{preference_str}, predict what is the list items to recommend to the user(u{user_id}). Please only answer the item IDs.",
                    "target": pos_str  # 这里修正了原代码的变量名错误
                }
                
                # --- 写入 JSONL ---
                # dump 将字典转为 json 字符串
                # ensure_ascii=False 保证如果以后有中文也能正常显示，虽然这里是ID
                json.dump(str_out, f_output, ensure_ascii=False)
                # 换行，这是 JSONL 的关键
                f_output.write("\n")

    print(f"处理完成！数据已保存至: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/data/UIO-LLM-SBR/mydatasets/', help='path of datasets')
    parser.add_argument('--dataset', default='diginetica', help='Tmall/diginetica/retailRocket_DSAN/Nowplaying') 
    parser.add_argument('--user_id_aug_size', type=int, default=1, help='augmentation sequence per usid')
    parser.add_argument('--target_aug_ratio', type=float, default=0.1, help='Target augmentation ratio (e.g., 0.1 for 10%)')
    opt = parser.parse_args()

    base_dir = os.path.join(opt.path, opt.dataset)
    train_file = os.path.join(base_dir, 'train.txt')
    # test_file = os.path.join(base_dir, 'test.txt')
    
    print(f"正在读取文件: {train_file}")
    # print(f"正在读取文件: {test_file}")

    try:
        # 加载 Pickle 数据
        train = pickle.load(open(train_file, 'rb'))
        # test = pickle.load(open(test_file, 'rb'))
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {train_file}")
        return 
    
    train_x = train[0]
    train_y = train[1]

    # test_x = test[0]
    # test_y = test[1] 

    # 原数据集总样本量 (Session级别)
    total_train_samples = len(train_x)
    print(f"训练集原始样本数量: {total_train_samples}")

    # --- 1. 识别用户 ---
    print("正在识别训练集用户...")
    user_session_train_dict = identify_sbr_users(train_x, train_y)
    total_users = len(user_session_train_dict)
    print(f"识别出独立用户数: {total_users}")

    # --- 2. 计算采样概率 ---
    # 目标生成的增强样本数
    target_aug_count = int(total_train_samples * opt.target_aug_ratio)
    
    # 如果不做采样，默认会生成的数量 (每个用户生成 user_id_aug_size 条)
    potential_aug_count = total_users * opt.user_id_aug_size
    
    if potential_aug_count > target_aug_count:
        # 需要降采样：计算保留概率
        keep_prob = target_aug_count / potential_aug_count
        print(f"【降采样模式】目标比例 {opt.target_aug_ratio*100}% (约 {target_aug_count} 条)。")
        print(f"当前潜在生成量 {potential_aug_count} 条，将以 {keep_prob:.4f} 的概率抽取用户。")
    else:
        # 不需要降采样（甚至可能不够，但这里我们只做上限控制，不强制增多）
        keep_prob = 1.0
        print(f"【全量模式】目标量 {target_aug_count} > 潜在量 {potential_aug_count}，将保留所有用户。")

    print("-" * 30)

    # --- 3. 生成数据 (传入 keep_prob) ---
    generate_sbr_jsonl(
        user_session_train_dict, 
        opt.user_id_aug_size, 
        os.path.join(base_dir, f'train_LLM4IDRec_size_{opt.user_id_aug_size}_rate_{opt.target_aug_ratio}.jsonl'),
        keep_prob=keep_prob
    )

        
    # print(f"测试集样本数量: {len(test_x)}")

    # # --- 执行识别 ---
    # print("正在通过相邻性还原测试集用户...")
    # user_session_test_dict = identify_sbr_users(test_x, test_y)

    # print(f"\n识别完成！总共识别出测试集的 {len(user_session_test_dict)} 个 独立用户。")
    # print("-" * 30)

    # generate_sbr_jsonl(user_session_test_dict, opt.user_id_aug_size, os.path.join(base_dir, 'test_LLM4IDRec.jsonl'))


if __name__ == "__main__":
    main()
