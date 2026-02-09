import random
import json
import pdb
import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import re

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
                grouped_data[f"u{user_count}"] = current_user_data
                user_count += 1
                current_user_data = [] # 重置
        else:
            # 循环结束，保存最后积累的数据
            grouped_data[f"u{user_count}"] = current_user_data

    return grouped_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/data/UIO-LLM-SBR/mydatasets/', help='path of datasets')
    parser.add_argument('--dataset', default='diginetica', help='Tmall/diginetica/retailrocket')
    parser.add_argument('--idicator', type=str, default="LLM4IDRec", help='method')
    parser.add_argument('--user_id_aug_size', type=int, default=1, help='augmentation sequence per usid')
    parser.add_argument('--target_aug_ratio', type=float, default=0.1, help='Target augmentation ratio (e.g., 0.1 for 10%)')
    opt = parser.parse_args()

    base_dir = os.path.join(opt.path, opt.dataset)
    raw_train_file = os.path.join(base_dir, f'train.txt')
    augmented_train_file = os.path.join(base_dir, f'train_augmented_{opt.idicator}_size_{opt.user_id_aug_size}_rate_{opt.target_aug_ratio}.txt')
    
    print(f"正在写入文件: {augmented_train_file}")

    raw_train = pickle.load(open(raw_train_file, 'rb'))

    n_node = {
        "diginetica": 43097,
        "Tmall": 40727,
        "Nowplaying": 60416,
        "retailRocket_DSAN": 36968
    }
    item_num = n_node[opt.dataset]

    raw_train_x = raw_train[0]
    raw_train_y = raw_train[1]

    user_session_raw_train_dict = identify_sbr_users(raw_train_x, raw_train_y)


    user_session_augmented_train_dict = dict()
    path_llm = f'{opt.path}/{opt.dataset}/{opt.idicator}_size_{opt.user_id_aug_size}_rate_{opt.target_aug_ratio}_predictions.json'
    cnt = 0
    with open(path_llm, "r") as f:
        for line in tqdm(f.readlines()):
            # pdb.set_trace()
            # if cnt > 5: break
            # else: cnt += 1
            example = json.loads(line.strip())
            # print(f"example:{example}")
            prompt = example["prompt"]
            prediction = example["prediction"]
            # print()
            # print(f"unique predictions:{set(prediction.split(',')[1:-1])}")
            pattern = r"items:(.*?), predict"
            match = re.search(pattern, prompt)
            if match:
                items_str = match.group(1).strip()
                clicked_item_list = items_str.split(',')
                clicked_item_list = [int(item[1:]) for item in clicked_item_list] # 去除item标识的i
                # print()
                # print(f"clicked_item_list:{clicked_item_list}")
                # print()
            else:
                # 如果正则没匹配到，跳过这条，防止使用了旧变量
                continue
            s_p = prompt.find('user(u')+6
            e_p = prompt.find(')')
            user_id = prompt[s_p-1:e_p] 
            # print(f"user_id:{user_id}")
            # print()
            
            # prediction 只有一个的时候
            try:
                item_id = int(prediction.strip()[1:]) # 去除item标识的i
                # 丢弃不合格id
                if item_id > item_num or item_id < 0:
                    continue
            except:
                print(f"error prediction: {prediction}")
                continue

            # 增强的序列根据userid增加到原来的dict中
            if user_id in user_session_augmented_train_dict:
                user_session_augmented_train_dict[user_id].append([clicked_item_list, item_id])  
            else:
                user_session_augmented_train_dict[user_id] = [[clicked_item_list, item_id]]
            ###################################################################

            # prediction 为多个的时候
            # llm_augmented_item_lists = set(prediction.split(',')[1:-1])
            # # 丢弃
            # if len(llm_augmented_item_lists) < 2:
            #     continue
            # new_llm_augmented_item_lists = [] 
            # for item in llm_augmented_item_lists:
            #     try:
            #         item_id = int(item.strip()[1:]) # 去除item标识的i
            #         # 丢弃不合格id
            #         if item_id > item_num or item_id < 0:
            #             continue
            #     except:
            #         print(f"error item id: {item}")
            #         continue
            #     new_llm_augmented_item_lists.append(item_id)
            # llm_augmented_item_lists = new_llm_augmented_item_lists[:]
            # # 增强的序列根据userid增加到原来的dict中
            # for item in llm_augmented_item_lists:
            #     if user_id in user_session_augmented_train_dict:
            #         user_session_augmented_train_dict[user_id].append([clicked_item_list, item])  
            #     else:
            #         user_session_augmented_train_dict[user_id] = [[clicked_item_list, item]]
            ###################################################################

    # ==========================================
    # 2. 合并数据 & 恢复顺序
    # ==========================================
    print("正在合并数据并恢复顺序...")
    
    final_x = []
    final_y = []
    
    augmented_count = 0
    
    # user_session_raw_train_dict 是由 list 生成的，Python 3.7+ 字典保证插入顺序
    # 遍历原始字典的 key，就能保证顺序与原始 train.txt 一致
    for raw_userid, raw_samples in user_session_raw_train_dict.items():
        
        # 1. 先加入原始数据
        # raw_samples 是 [[x,y], [x,y]...]
        for sample in raw_samples:
            final_x.append(sample[0])
            final_y.append(sample[1])
            
        # 2. 检查该用户是否有增强数据，如果有，追加到后面
        if raw_userid in user_session_augmented_train_dict:
            aug_samples = user_session_augmented_train_dict[raw_userid]
            for sample in aug_samples:
                final_x.append(sample[0])
                final_y.append(sample[1])
                augmented_count += 1

    print(f"原始数据处理完毕。")
    print(f"共增加了 {augmented_count} 条增强样本。")
    print(f"最终数据集总大小: {len(final_x)}")

    # ==========================================
    # 3. 保存文件 (Pickle)
    # ==========================================
    final_data = (final_x, final_y)
    
    print(f"正在写入文件: {augmented_train_file}")
    with open(augmented_train_file, 'wb') as f:
        pickle.dump(final_data, f)
        
    print("写入完成！")

if __name__ == "__main__":
    main()
