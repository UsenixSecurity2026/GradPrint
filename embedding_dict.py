import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def extract_embeddings(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    target_layer_idx: int = 4,
    batch_size: int = 1024,
    device: str = None
):
    """
    提取模型的嵌入向量
    
    Args:
        model (torch.nn.Module): 预训练的PyTorch模型
        x_train (np.ndarray): 输入数据，形状为 [num_samples, input_dim]
        target_layer_idx (int): 目标层的索引（默认第4层）
        batch_size (int): 批处理大小（默认64）
        device (str): 计算设备（'cpu'或'cuda'，默认自动检测）
    
    Returns:
        np.ndarray: 嵌入向量，形状为 [num_samples, embedding_dim]
    """
    class EmbeddingExtractor:
        def __init__(self, model, target_layer):
            self.model = model
            self.embedding = None
            self.hook = model[target_layer].register_forward_hook(self._hook_fn)
        
        def _hook_fn(self, module, input, output):
            self.embedding = output.detach()
        
        def remove_hook(self):
            self.hook.remove()

    # # 设备自动检测
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    # 数据预处理
    if not isinstance(x_train, torch.Tensor):
        x_tensor = torch.from_numpy(x_train).float()  # 共享内存转换
    else:
        x_tensor = x_train.clone().detach().float()  # 复制数据并分离梯度
    
    if not isinstance(y_train, torch.Tensor):
        y_tensor = torch.from_numpy(y_train).long()  # 共享内存转换
    else:
        y_tensor = y_train.clone().detach().long()  # 复制数据并分离梯度
    
    # 移动到指定设备
    model = model.to(device).eval()
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)
    
    # 创建数据加载器
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化提取器
    extractor = EmbeddingExtractor(model, target_layer_idx)
    
    # 存储结果
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)  # 触发前向传播
            embeddings.append(extractor.embedding)
            #embeddings.append(extractor.embedding.cpu().numpy())
            labels.append(batch_labels.cpu())
    
    # 清理资源
    extractor.remove_hook()
    
    # 合并结果
    return torch.cat(embeddings, dim=0).cpu(), torch.cat(labels, dim=0).cpu()
    #return np.concatenate(embeddings, axis=0)