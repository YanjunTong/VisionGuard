import torch
import numpy as np
import os
import cv2
from PIL import Image
import clip


VIDEO_DIR = "./"              # 测试视频文件夹
OUTPUT_FILE = "submission.txt"          # 提交结果文件
MODEL_PATH = "saved_models/model_epoch_210.pth"  # 训练好的模型路径
TEXT_QUERY = ["检测打斗", "检测火灾", "检测摔倒", "检测车辆碰撞"]  # 测试文本
CLIP_FRAME_NUM = 16
CLIP_STRIDE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORY_TO_ID = {
    "正常": 0,
    "打斗": 1,
    "火灾": 2,
    "摔倒": 3,
    "车辆碰撞": 4
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}

# ========== 模型定义 ==========
class CrossModalModel(torch.nn.Module):
    def __init__(self, video_dim=512, text_dim=512, num_categories=len(CATEGORY_TO_ID)):
        super().__init__()
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(video_dim + text_dim + 1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
        )
        self.attn = torch.nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.anomaly_head = torch.nn.Linear(256, 1)
        self.category_head = torch.nn.Linear(256, num_categories)
        self.loc_head = torch.nn.Linear(256, 2)  # 定位头：预测偏移

    def forward(self, video_feat, text_feats, sim_score):
        text_feat = text_feats.mean(dim=1)  # [B, text_dim]
        fused = torch.cat([video_feat, text_feat, sim_score], dim=1)
        fused = self.fusion(fused).unsqueeze(1)  # [B, 1, 256]
        attn_out, _ = self.attn(fused, fused, fused)
        attn_out = attn_out.squeeze(1)
        return self.anomaly_head(attn_out), self.category_head(attn_out), self.loc_head(attn_out)

# ========== 加载模型 ==========
model = CrossModalModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========== 加载CLIP ==========
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

# ========== 提取视频clip特征 ==========
def extract_video_clip_feats(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(clip_preprocess(pil_img).unsqueeze(0))
    cap.release()

    feats = []
    frame_mapping = []
    for i in range(0, len(frames) - CLIP_FRAME_NUM + 1, CLIP_STRIDE):
        clip_imgs = torch.cat(frames[i:i+CLIP_FRAME_NUM], dim=0).to(DEVICE)
        with torch.no_grad():
            f = clip_model.encode_image(clip_imgs).mean(dim=0)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        frame_mapping.append([i, i + CLIP_FRAME_NUM - 1])
    return np.array(feats), np.array(frame_mapping), len(frames)

# ========== 提取文本特征 ==========
def extract_text_feats(text_list):
    text_tokens = clip.tokenize(text_list).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_text(text_tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

text_feats_np = extract_text_feats(TEXT_QUERY)

# ========== 推理 ==========
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for video_file in os.listdir(VIDEO_DIR):
        if not video_file.endswith((".mp4", ".avi")):
            continue
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(VIDEO_DIR, video_file)

        # 提取视频特征
        video_feats, frame_mapping, total_frames = extract_video_clip_feats(video_path)
        if len(video_feats) == 0:
            f_out.write(f"{video_id} -1 -1 正常\n")
            continue

        # 计算相似度
        sim_matrix = video_feats @ text_feats_np.T  
        sim_scores = sim_matrix.mean(axis=1, keepdims=True)  

        # 转Tensor
        video_feats = torch.FloatTensor(video_feats).to(DEVICE)
        text_feats = torch.FloatTensor(text_feats_np).unsqueeze(0).repeat(len(video_feats), 1, 1).to(DEVICE)
        sim_scores = torch.FloatTensor(sim_scores).to(DEVICE)

        with torch.no_grad():
            anomaly_logits, category_logits, loc_offsets = model(video_feats, text_feats, sim_scores)

        anomaly_probs = torch.sigmoid(anomaly_logits).cpu().numpy().flatten()
        category_preds = torch.argmax(category_logits, dim=1).cpu().numpy()
        loc_offsets = loc_offsets.cpu().numpy()

        # 生成结果
        results = []
        for i in range(len(video_feats)):
            if anomaly_probs[i] > 0.5:  # 异常阈值
                start = max(0, int(frame_mapping[i, 0] + loc_offsets[i, 0]))
                end = min(int(frame_mapping[i, 1] + loc_offsets[i, 1]), total_frames - 1)
                category = ID_TO_CATEGORY[category_preds[i]]
                results.append([start, end, category])

        # 合并重叠片段
        results.sort(key=lambda x: x[0])
        merged = []
        for res in results:
            if not merged:
                merged.append(res)
            else:
                last = merged[-1]
                if res[0] <= last[1] and res[2] == last[2]:
                    merged[-1] = [last[0], max(last[1], res[1]), last[2]]
                else:
                    merged.append(res)

        # 输出
        if merged:
            for m in merged:
                f_out.write(f"{video_id} {m[0]} {m[1]} {m[2]}\n")
        else:
            f_out.write(f"{video_id} -1 -1 正常\n")

print(f"推理完成！结果已保存到 {OUTPUT_FILE}")