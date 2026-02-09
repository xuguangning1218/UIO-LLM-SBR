import pickle
import json
import os
from tqdm import tqdm
import argparse

# ================= 配置区域 =================
parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('--path', default='/data/UIO-LLM-SBR/datasets/', help='path of datasets')
parser.add_argument('--dataset', default='Tmall', help='Tmall/diginetica/retailrocket') 

opt = parser.parse_args()
# ===========================================

def generate_prompt_and_target(session_list, target_item):
    """
    根据 LLM4IDRec 论文构建 Prompt
    User ID 统一使用 "anonymous"
    """
    # 1. 设置统一的用户 ID
    user_id = "anonymous"
    
    # 2. 将 Session 列表转换为字符串 (例如: "3005, 30012")
    # 确保 item 是字符串格式
    history_str = ", ".join([str(item) for item in session_list])
    
    # 3. 构建符合论文要求的 Template 
    # Template: Input: Given the user [UID]'s clicked list items: [History], predict what are items to recommend to the user [UID]. Please only answer the items.\nOutput: 
    template = "Input: Given the user {}'s clicked list items: {}, predict what are items to recommend to the user {}. Please only answer the items.\nOutput: "
    
    prompt = template.format(user_id, history_str, user_id)
    target = str(target_item)
    
    return prompt, target

def save_to_jsonl(data_x, data_y, output_path):
    """
    将处理后的数据保存为 .jsonl 格式
    """
    print(f"正在转换数据并保存到: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 zip 同时遍历 session 和 target
        for session, target_item in tqdm(zip(data_x, data_y), total=len(data_x)):
            
            prompt, target = generate_prompt_and_target(session, target_item)
            
            # 构建 JSON 对象
            line_data = {
                "prompt": prompt,
                "target": target
            }
            
            # 写入文件
            f.write(json.dumps(line_data, ensure_ascii=False) + '\n')

def main():
    # 1. 构建文件路径
    base_dir = opt.path + opt.dataset
    train_file = os.path.join(base_dir, 'train.txt')
    test_file = os.path.join(base_dir, 'test.txt')

    print(f"正在读取文件: {train_file}")
    print(f"正在读取文件: {test_file}")

    # 2. 加载 Pickle 数据 (你提供的读取逻辑)
    try:
        train = pickle.load(open(train_file, 'rb'))
        test = pickle.load(open(test_file, 'rb'))
    except FileNotFoundError:
        print("错误：找不到输入文件，请检查 opt.path 和 opt.dataset 配置是否正确。")
        return

    # 3. 提取 x 和 y
    train_x = train[0]
    train_y = train[1]
    
    test_x = test[0]
    test_y = test[1]

    print(f"训练集数量: {len(train_x)}")
    print(f"测试集数量: {len(test_x)}")

    # 4. 转换并保存为 JSONL
    # 保存路径: opt.path + opt.dataset + '/train.jsonl'
    train_output_path = os.path.join(base_dir, 'train_LLM4IDRec.jsonl')
    test_output_path = os.path.join(base_dir, 'test_LLM4IDRec.jsonl')
    save_to_jsonl(train_x, train_y, train_output_path)
    save_to_jsonl(test_x, test_y, test_output_path)

    print("\n所有转换完成！")
    print(f"生成的训练集: {train_output_path}")
    print(f"生成的训练集: {test_output_path}")

if __name__ == "__main__":
    main()