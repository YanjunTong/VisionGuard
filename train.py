import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ==================== 配置部分 ====================
VIDEO_FEAT_DIR = "preprocessed_data/video_features"
TEXT_FEAT_DIR  = "preprocessed_data/text_features"
SIM_MATRIX_DIR = "preprocessed_data/sim_matrices"
PSEUDO_LABEL_DIR = "pseudo_labels"  # 伪标签文件夹
SAVE_MODEL_DIR = "saved_models"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

# 类别映射
CATEGORY_TO_ID = {
    "正常": 0,
    "打斗": 1,
    "火灾": 2,
    "摔倒": 3,
    "车辆碰撞": 4
}

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 500  # 你的训练轮数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH_SAVE = 10  # 你的模型保存频率（每10轮存一次）

# ==================== Dataset 类（加载伪标签） ====================
class AnomalyDataset(Dataset):
    def __init__(self, video_ids, label_dict):
        self.samples = []
        for vid in video_ids:
            # 加载特征
            video_feats = np.load(os.path.join(VIDEO_FEAT_DIR, f"{vid}_feats.npy"))
            text_feats  = np.load(os.path.join(TEXT_FEAT_DIR, f"{vid}_text_feats.npy"))
            sim_matrix  = np.load(os.path.join(SIM_MATRIX_DIR, f"{vid}_sim_matrix.npy"))
            frame_mapping = np.load(os.path.join(VIDEO_FEAT_DIR, f"{vid}_frame_mapping.npy"))
            
            # 视频标签
            category = label_dict[vid]["category"]
            category_id = CATEGORY_TO_ID[category]
            label = 0 if category == "正常" else 1  # 0=正常, 1=异常
            
            # 加载伪标签（每个视频一个txt文件）
            pseudo_label_path = os.path.join(PSEUDO_LABEL_DIR, f"{vid}.txt")
            events = []
            if os.path.exists(pseudo_label_path):
                with open(pseudo_label_path, "r") as f:
                    for line in f:
                        start, end = map(int, line.strip().split())
                        events.append([start, end])
            
            # 每个clip一个样本
            for i in range(len(video_feats)):
                clip_start, clip_end = frame_mapping[i]
                start_offset = 0
                end_offset = 0
                # 检查当前clip是否与异常事件重叠
                for (gt_start, gt_end) in events:
                    if not (clip_end < gt_start or clip_start > gt_end):
                        # 计算偏移
                        start_offset = gt_start - clip_start
                        end_offset = gt_end - clip_end
                        break  # 只匹配第一个重叠事件
                
                self.samples.append({
                    "video_feat": video_feats[i],
                    "text_feats": text_feats,
                    "sim_score": sim_matrix[i].mean(),
                    "label": label,
                    "category_id": category_id,
                    "frame_range": frame_mapping[i],
                    "start_offset": start_offset,
                    "end_offset": end_offset
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.FloatTensor(s["video_feat"]),
            torch.FloatTensor(s["text_feats"]),
            torch.FloatTensor([s["sim_score"]]),
            torch.LongTensor([s["label"]]),
            torch.LongTensor([s["category_id"]]),
            torch.LongTensor(s["frame_range"]),
            torch.FloatTensor([s["start_offset"]]),
            torch.FloatTensor([s["end_offset"]])
        )

# ==================== 模型定义 ====================
class CrossModalModel(nn.Module):
    def __init__(self, video_dim=512, text_dim=512, num_categories=len(CATEGORY_TO_ID)):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + text_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        self.anomaly_head = nn.Linear(256, 1)
        self.category_head = nn.Linear(256, num_categories)
        self.loc_head = nn.Linear(256, 2)  # 预测起始/结束帧偏移

    def forward(self, video_feat, text_feats, sim_score):
        text_feat = text_feats.mean(dim=1)  # [B, text_dim]
        fused = torch.cat([video_feat, text_feat, sim_score], dim=1)
        fused = self.fusion(fused).unsqueeze(1)  # [B, 1, 256]
        attn_out, _ = self.attn(fused, fused, fused)
        attn_out = attn_out.squeeze(1)
        return self.anomaly_head(attn_out), self.category_head(attn_out), self.loc_head(attn_out)

# ==================== 训练函数 ====================
def train():
    # 视频ID和标签（保留你的所有视频，包括正常视频）
    label_dict = {
        "car 01": {"category": "车辆碰撞"},
        "car 02": {"category": "车辆碰撞"},
        "car 03": {"category": "车辆碰撞"},
        "car 04": {"category": "车辆碰撞"},
        "car 05": {"category": "车辆碰撞"},
        "car 06": {"category": "车辆碰撞"},
        "car 07": {"category": "车辆碰撞"},
        "car 08": {"category": "车辆碰撞"},
        "car 10": {"category": "车辆碰撞"},
        "car 12": {"category": "车辆碰撞"},
        "normal_1": {"category": "正常"},
        "normal_2": {"category": "正常"},
        "normal_3": {"category": "正常"},
        "normal_4": {"category": "正常"},
        "normal_5": {"category": "正常"},
        "normal_6": {"category": "正常"},
        "normal_7": {"category": "正常"},
        "normal_8": {"category": "正常"},
        "normal_9": {"category": "正常"},
        "normal_10": {"category": "正常"},
    }
    video_ids = list(label_dict.keys())
    
    dataset = AnomalyDataset(video_ids, label_dict)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CrossModalModel().to(DEVICE)
    criterion_anomaly = nn.BCEWithLogitsLoss()
    criterion_category = nn.CrossEntropyLoss()
    criterion_loc = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in dataloader:
            video_feat, text_feats, sim_score, label, category_id, frame_range, start_offset, end_offset = [
                x.to(DEVICE) for x in batch
            ]
            
            optimizer.zero_grad()
            
            anomaly_logit, category_logit, loc_offset = model(video_feat, text_feats, sim_score)
            
            # 异常二分类损失（无修改）
            loss_anomaly = criterion_anomaly(anomaly_logit, label.float())
            
            # -------------------------- 关键修改1：异常类别损失 --------------------------
            # 用 flatten() 替代 squeeze()，避免维度丢失导致 batch_size 0
            mask = (label == 1).squeeze()
            if mask.float().sum() > 0:  # 确保有异常样本
                loss_category = criterion_category(
                    category_logit[mask],
                    category_id[mask].flatten()  # 这里从 squeeze() 改成 flatten()
                )
            else:
                loss_category = torch.tensor(0.0).to(DEVICE)
            
            # -------------------------- 关键修改2：定位损失 --------------------------
            # 同样用 flatten() 替代 squeeze()，确保维度一致
            loss_loc = torch.tensor(0.0).to(DEVICE)
            if mask.float().sum() > 0:
                # 所有偏移预测都用 flatten() 保证一维张量
                pred_start_offset = loc_offset[mask, 0].flatten()
                pred_end_offset = loc_offset[mask, 1].flatten()
                gt_start_offset = start_offset[mask].flatten()
                gt_end_offset = end_offset[mask].flatten()
                
                loss_loc = criterion_loc(
                    torch.stack([pred_start_offset, pred_end_offset], dim=1),
                    torch.stack([gt_start_offset, gt_end_offset], dim=1)
                )
            
            # 总损失计算（无修改）
            loss = loss_anomaly + loss_category + 0.5 * loss_loc
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印日志（无修改）
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
        # 按频率保存模型（保留你的逻辑）
        if (epoch + 1) % EPOCH_SAVE == 0:
            save_path = os.path.join(SAVE_MODEL_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型已保存：{save_path}")

if __name__ == "__main__":
    train()