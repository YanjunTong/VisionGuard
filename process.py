import torch
import clip
import cv2
from PIL import Image
import numpy as np
import os
from typing import List, Tuple

# -------------------------- 1. 基础配置 --------------------------
class Config:
    # 模型参数
    MODEL_NAME = "ViT-B/32"  # CLIP 模型类型，也可换 ViT-L/14
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_FRAME_NUM = 16  # 每个视频clip包含的帧数
    CLIP_STRIDE = 8      # 滑动步长
    # 保存路径
    VIDEO_FEAT_SAVE_DIR = "preprocessed_data/video_features"  # 视频特征保存路径
    TEXT_FEAT_SAVE_DIR = "preprocessed_data/text_features"    # 文本特征保存路径
    SIM_MATRIX_SAVE_DIR = "preprocessed_data/sim_matrices"    # 相似度矩阵保存路径
    # 创建保存目录
    for dir_path in [VIDEO_FEAT_SAVE_DIR, TEXT_FEAT_SAVE_DIR, SIM_MATRIX_SAVE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

# 加载CLIP模型（
model, preprocess = clip.load(Config.MODEL_NAME, device=Config.DEVICE)


# -------------------------- 2. 视频特征提取（含帧范围映射） --------------------------
def extract_video_clip_feat(
    video_path: str
) -> Tuple[np.ndarray, np.ndarray]:

    cap = cv2.VideoCapture(video_path)
    frame_list = []  
    frame_idx = 0    

    # 1. 读取并预处理所有帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # BGR转RGB，再用CLIP预处理
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(pil_img).unsqueeze(0)  
        frame_list.append(img_tensor)
        frame_idx += 1
    cap.release()

    # 2. 按clip分割并提取特征
    clip_num = max(1, len(frame_list) - Config.CLIP_FRAME_NUM + 1)  # clip总数
    video_feats = []
    clip_frame_mapping = []

    for i in range(clip_num):
        # 取当前clip的帧（滑动窗口）
        start_frame = i * Config.CLIP_STRIDE
        end_frame = start_frame + Config.CLIP_FRAME_NUM
        # 处理最后一个clip（避免帧不足）
        if end_frame > len(frame_list):
            end_frame = len(frame_list)
            start_frame = max(0, end_frame - Config.CLIP_FRAME_NUM)
        
        # 拼接帧并提取特征
        clip_imgs = torch.cat(frame_list[start_frame:end_frame], dim=0).to(Config.DEVICE)
        with torch.no_grad():
            clip_feat = model.encode_image(clip_imgs)  # shape=(frame_in_clip, 512)
        # 平均池化得到clip级特征，再L2归一化（跨模态匹配必备）
        clip_feat = clip_feat.mean(dim=0)  # shape=(512,)
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)  # 归一化
        
        # 保存结果
        video_feats.append(clip_feat.cpu().numpy())
        clip_frame_mapping.append([start_frame, end_frame - 1])  # 帧索引从0开始，结束帧减1对齐实际帧

    # 转为numpy数组
    return np.array(video_feats), np.array(clip_frame_mapping)


# -------------------------- 3. 文本特征提取（支持增强文本） --------------------------
def extract_text_feat(
    text_list: List[str]
) -> np.ndarray:
    """
    提取文本描述的特征（支持原始文本+增强文本，应对未知测试文本）
    Args:
        text_list: 文本描述列表（如 ["检测火灾", "视频中有火焰", "识别火灾事件"]）
    Returns:
        text_feats: 文本特征，shape=(text_num, 512)
    """
    # CLIP文本预处理（自动添加<|startoftext|>和<|endoftext|>）
    text_tokens = clip.tokenize(text_list).to(Config.DEVICE)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)  # shape=(text_num, 512)
    # L2归一化（与视频特征保持一致）
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats.cpu().numpy()


# -------------------------- 4. 跨模态相似度计算（核心匹配逻辑） --------------------------
def calculate_cross_modal_sim(
    video_feats: np.ndarray,
    text_feats: np.ndarray
) -> np.ndarray:
    """
    计算视频clip特征与文本特征的余弦相似度（值越大，匹配度越高）
    Args:
        video_feats: 视频clip特征，shape=(clip_num, 512)
        text_feats: 文本特征，shape=(text_num, 512)
    Returns:
        sim_matrix: 相似度矩阵，shape=(clip_num, text_num)，sim_matrix[i][j]是第i个clip与第j个文本的相似度
    """
    # 余弦相似度 = 特征点积（因已归一化，点积=余弦值）
    sim_matrix = np.matmul(video_feats, text_feats.T)  # shape=(clip_num, text_num)
    return sim_matrix


# -------------------------- 5. 整体Pipeline调用入口 --------------------------
def run_preprocess_pipeline(
    video_dir: str,
    text_desc_dict: dict  # key: 视频名（如"3"），value: 该视频的文本描述列表（含增强文本）
) -> None:
    """
    批量处理视频文件夹中的所有视频，完成全流程预处理并保存结果
    Args:
        video_dir: 视频文件夹路径（如 "train_videos"）
        text_desc_dict: 视频-文本映射字典（如 {"3": ["检测打斗", "视频中有人打斗"]}）
    """
    # 遍历视频文件夹中的所有视频
    for video_filename in os.listdir(video_dir):
        if not video_filename.endswith((".mp4", ".avi")):  # 过滤非视频文件
            continue
        
        # 1. 解析视频信息
        video_name = os.path.splitext(video_filename)[0]  # 视频名（如"3"，对应video_id）
        video_path = os.path.join(video_dir, video_filename)
        print(f"正在处理视频：{video_name}")

        # 2. 提取视频特征和帧映射
        video_feats, clip_frame_mapping = extract_video_clip_feat(video_path)
        # 保存视频相关结果
        np.save(
            os.path.join(Config.VIDEO_FEAT_SAVE_DIR, f"{video_name}_feats.npy"),
            video_feats
        )
        np.save(
            os.path.join(Config.VIDEO_FEAT_SAVE_DIR, f"{video_name}_frame_mapping.npy"),
            clip_frame_mapping
        )

        # 3. 提取文本特征（从text_desc_dict获取该视频的文本）
        if video_name not in text_desc_dict:
            print(f"警告：视频{video_name}无对应文本描述，跳过文本处理")
            continue
        text_list = text_desc_dict[video_name]
        text_feats = extract_text_feat(text_list)
        # 保存文本特征
        np.save(
            os.path.join(Config.TEXT_FEAT_SAVE_DIR, f"{video_name}_text_feats.npy"),
            text_feats
        )

        # 4. 计算跨模态相似度矩阵
        sim_matrix = calculate_cross_modal_sim(video_feats, text_feats)
        # 保存相似度矩阵
        np.save(
            os.path.join(Config.SIM_MATRIX_SAVE_DIR, f"{video_name}_sim_matrix.npy"),
            sim_matrix
        )

        print(f"视频{video_name}预处理完成！\n")


# -------------------------- 6. 测试示例（直接运行即可验证） --------------------------
if __name__ == "__main__":
    # 示例1：视频文件夹路径（请替换为你的训练集/测试集视频路径）
    TEST_VIDEO_DIR = "./video"  # 假设文件夹下有 "3.mp4" "5.mp4" 等视频
    # 示例2：视频-文本映射字典（key=视频名，value=原始文本+增强文本）
    TEST_TEXT_DESC_DICT = {
        "car 01": ["检测撞击", "视频中有车辆出现碰撞", "识别交通事故", "发现视频中发生交通事故"],
        "car 02": ["发现追尾事故", "侦测到车辆剐蹭", "识别车辆碰撞", "视频中存在车辆撞击事件"],
        "car 03": ["分析交通意外", "行人与电动车发生碰撞", "发现车辆侧翻", "识别道路交通事故"],
        "car 04": ["检测交通事故", "视频中有车辆出现碰撞", "识别交通事故", "发现视频中发生交通事故"],
        "car 05": ["发现追尾事故", "检测到两车发生追尾", "识别车辆碰撞", "视频中存在车辆撞击事件"],
        "car 06": ["检测交通事故", "视频中有车辆出现碰撞", "识别交通事故", "发现视频中发生交通事故"],
        "car 07": ["发现追尾事故", "侦测到车辆剐蹭", "识别车辆碰撞", "视频中存在车辆撞击事件"],
        "car 08": ["分析交通意外", "视频中有车辆出现碰撞", "发现车辆侧翻", "识别道路交通事故"],
        "car 10":["检测撞击", "视频中有车辆出现碰撞", "识别交通事故", "发现视频中发生交通事故"],
        "car 12":["发现追尾事故", "侦测到车辆剐蹭", "识别车辆碰撞", "视频中存在车辆撞击事件"],
        "normal_1":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_2":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_3":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_4":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_5":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_6":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_7":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_8":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_9":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"],
        "normal_10":["检测正常", "视频中无异常事件", "识别正常状态", "发现视频中没有异常"]
    }
    # 运行全流程预处理
    run_preprocess_pipeline(
        video_dir=TEST_VIDEO_DIR,
        text_desc_dict=TEST_TEXT_DESC_DICT
    )