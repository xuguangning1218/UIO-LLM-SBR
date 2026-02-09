import argparse
import os
import pickle
from tqdm import tqdm

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/data/UIO-LLM-SBR/datasets/', help='path of datasets')
    parser.add_argument('--dataset', default='diginetica', help='Tmall/diginetica/retailrocket') 
    opt = parser.parse_args()

    base_dir = os.path.join(opt.path, opt.dataset)
    train_file = os.path.join(base_dir, 'train.txt')
    
    print(f"正在读取文件: {train_file}")

    try:
        # 加载 Pickle 数据
        train = pickle.load(open(train_file, 'rb'))
        train_x = train[0]
        train_y = train[1]
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {train_file}")
        return 
    
    print(f"训练集样本数量: {len(train_x)}")

    # --- 执行识别 ---
    print("正在通过相邻性还原用户...")
    user_session_dict = identify_sbr_users(train_x, train_y)

    print(f"\n识别完成！总共识别出 {len(user_session_dict)} 个 独立用户。")
    print("-" * 30)
    
    # 打印前 5 个用户的详细数据作为检查
    count = 0
    for user, sessions in user_session_dict.items():
        print(f"=== {user} (包含 {len(sessions)} 条交互记录) ===")
        for sample in sessions:
            print(f"  Session: {sample[0]}, Target: {sample[1]}")
        print("")
        
        count += 1
        if count >= 5:
            break

if __name__ == "__main__":
    main()