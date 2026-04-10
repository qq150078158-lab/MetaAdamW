"""EarlyStopping"""

# ==================== 早停定义 ====================
class EarlyStopping:
    """早停类，当验证损失不再下降时停止训练"""

    def __init__(self, patience=2, min_delta=1e-2, restore_best_weights=True):
        """
        Args:
            patience: 验证损失连续不下降的epoch数阈值
            min_delta: 判定为改善的最小变化量
            restore_best_weights: 是否恢复最佳模型权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # 验证损失改善
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            # 验证损失未改善
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        return self.early_stop
