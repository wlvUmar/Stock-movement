import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import math
from typing import Optional, Tuple, Dict, Any

from utils import ModelConfig



class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for sequence modeling"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        return output


class ResidualBlock(nn.Module):
    """Residual block with layer norm"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.layer_norm(sublayer_output))



class AdvancedStockLSTM(nn.Module):
    """Advanced LSTM model for stock movement prediction with modern architecture"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        
        # LSTM layers
        lstm_hidden_size = config.hidden_size // 2 if config.bidirectional else config.hidden_size
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        if config.use_attention:
            self.attention = MultiHeadAttention(
                config.hidden_size, 
                num_heads=8, 
                dropout=config.dropout
            )
            
        if config.use_residual:
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(config.hidden_size, config.dropout) 
                for _ in range(2)
            ])
            
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size)
            
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.output_layers = self._build_output_layers()
        self._init_weights()
        

    def _build_output_layers(self) -> nn.Module:
        """Build output layers based on task type"""
        layers = []
        
        # Common layers
        layers.extend([
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(self.config.dropout)
        ])
        
        # Task-specific output layer
        if self.config.task_type == "classification":
            layers.extend([
                nn.Linear(self.config.hidden_size // 4, 1),
                nn.Sigmoid()
            ])
        elif self.config.task_type == "multi_class":
            layers.extend([
                nn.Linear(self.config.hidden_size // 4, self.config.num_classes),
                nn.Softmax(dim=1)
            ])
        else:  # regression
            layers.append(nn.Linear(self.config.hidden_size // 4, self.config.output_size))
            
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            mask: Optional attention mask
            
        Returns:
            Dictionary containing outputs and intermediate results
        """
        batch_size, seq_len, _ = x.size()
        x = self.input_projection(x)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.config.use_attention:
            attended = self.attention(lstm_out, mask)
            
            if self.config.use_residual:
                lstm_out = self.residual_blocks[0](lstm_out, attended)
            else:
                lstm_out = attended
        
        if self.config.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
            
        features = self.feature_extractor(lstm_out)
        
        if self.config.use_residual and len(self.residual_blocks) > 1:
            features = self.residual_blocks[1](lstm_out, features)
        
        # Global average pooling or use last time step
        if self.config.use_attention:
            # Weighted average based on attention
            pooled = torch.mean(features, dim=1)
        else:
            # Use last time step
            pooled = features[:, -1, :]
        
        # Output prediction
        output = self.output_layers(pooled)
        
        return {
            'output': output,
            'features': pooled,
            'sequence_output': features,
            'hidden_state': hidden,
            'cell_state': cell
        }
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities"""
        self.eval()
        with torch.no_grad():
            result = self.forward(x)
            if self.config.task_type == "multi_class":
                return result['output']
            elif self.config.task_type == "classification":
                probs = result['output']
                return torch.cat([1 - probs, probs], dim=1)
            else:
                return result['output']
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract attention weights for interpretability"""
        if not self.config.use_attention:
            return None
            
        self.eval()
        with torch.no_grad():
            x = self.input_projection(x)
            x = self.pos_encoding(x)
            lstm_out, _ = self.lstm(x)
            
            # Get attention weights (would need to modify attention module)
            # This is a placeholder - actual implementation would require
            # modifying the attention module to return weights
            return None


class StockLSTMTrainer:
    """Training utilities for the stock LSTM model"""
    
    def __init__(self, model: AdvancedStockLSTM, config: ModelConfig):
        self.model = model
        self.config = config
        
    def configure_optimizer(self, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
        """Configure optimizer with weight decay"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
    
    def configure_scheduler(self, optimizer: torch.optim.Optimizer, num_epochs: int, steps_per_epoch) -> torch.optim.lr_scheduler._LRScheduler:
        """Configure learning rate scheduler"""
        
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                     criterion: nn.Module) -> Dict[str, float]:
        """Single training step with gradient clipping"""
        x, y = batch
        
        # Forward pass
        outputs = self.model(x)
        predictions = outputs['output']
        
        # Calculate loss
        if self.config.task_type == "multi_class":
            loss = criterion(predictions, y.long())
        else:
            loss = criterion(predictions, y.float())
        
        # Backward pass with gradient clipping
        loss.backward()
        
        if self.config.gradient_clip > 0:
            clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # Calculate metrics
        metrics = {'loss': loss.item()}
        
        if self.config.task_type in ["classification", "multi_class"]:
            if self.config.task_type == "classification":
                pred_classes = (predictions > 0.5).float()
                accuracy = (pred_classes.squeeze() == y.float()).float().mean()
            else:
                pred_classes = predictions.argmax(dim=1)
                accuracy = (pred_classes == y.long()).float().mean()
            
            metrics['accuracy'] = accuracy.item()
        
        return metrics


def create_stock_lstm(input_size: int, 
                     task_type: str = "classification",
                     model_size: str = "medium") -> AdvancedStockLSTM:
    """
    Factory function to create stock LSTM models
    
    Args:
        input_size: Number of input features
        task_type: "classification", "multi_class", or "regression"
        model_size: "small", "medium", "large"
    """
    
    size_configs = {
        "small": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "use_attention": False,
            "use_residual": False,
            "use_layer_norm": False
        },
        "medium": {
            "hidden_size": 128,
            "num_layers": 3,
            "dropout": 0.3,
            "use_attention": True,
            "use_residual": True,
            "use_layer_norm": True
        },
        "large": {
            "hidden_size": 256,
            "num_layers": 4,
            "dropout": 0.4,
            "use_attention": True,
            "use_residual": True,
            "use_layer_norm": True
        }
    }
    
    config = ModelConfig(
        input_size=input_size,
        task_type=task_type,
        **size_configs[model_size]
    )
    
    return AdvancedStockLSTM(config)


# Usage examples
if __name__ == "__main__":
    config = ModelConfig(
        input_size=20,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        bidirectional=True,
        use_attention=True,
        task_type="classification"
    )
    
    model = AdvancedStockLSTM(config)
    
    # Example input
    batch_size, seq_len, input_size = 32, 60, 20
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    outputs = model(x)
    print(f"Output shape: {outputs['output'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    
    # Create different model sizes
    small_model = create_stock_lstm(input_size=20, model_size="small")
    medium_model = create_stock_lstm(input_size=20, model_size="medium")
    large_model = create_stock_lstm(input_size=20, model_size="large")
    
    print(f"Small model parameters: {sum(p.numel() for p in small_model.parameters()):,}")
    print(f"Medium model parameters: {sum(p.numel() for p in medium_model.parameters()):,}")
    print(f"Large model parameters: {sum(p.numel() for p in large_model.parameters()):,}")