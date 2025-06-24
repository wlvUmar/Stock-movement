import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from .dataloader import create_stock_datasets, create_data_loaders
from .model import create_stock_lstm, StockLSTMTrainer
from utils import DatasetConfig
from logger_setup import setup_logging

setup_logging("logs/ml.log")

def train_stock_model(csv_path="temp.csv",  epochs=50,  batch_size=32,  learning_rate=1e-3, model_save_path="stock_model.pth", incremental=False, existing_model_path=None):
    """
    Train stock prediction model (full retraining or incremental)
    
    Args:
        csv_path: Path to your CSV file
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        model_save_path: Where to save the trained model
        incremental: If True, continue training existing model
        existing_model_path: Path to existing model for incremental training
    
    Returns:
        Trained model
    """
    
    logger = logging.getLogger(__name__)
    
    config = DatasetConfig(
        window_size=60,          
        horizon=10,               
        target_type="classification", 
        scaling_method="robust", 
        validation_split=0.2,
        test_split=0.1
    )
    
    logger.info("Loading and preparing data...")
    train_ds, val_ds, test_ds = create_stock_datasets(csv_path, config)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )
    
    input_size = len(train_ds.feature_names)
    
    if incremental and existing_model_path:
        logger.info("Loading existing model for incremental training...")
        model_data = load_trained_model(existing_model_path)
        model = model_data['model']
        
        trainer = StockLSTMTrainer(model, model.config)
        optimizer = trainer.configure_optimizer(learning_rate)
        
        try:
            checkpoint = torch.load(existing_model_path, map_location='cpu')
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state for incremental training")
        except:
            logger.warning("Could not load optimizer state, using fresh optimizer")
    else:
        model = create_stock_lstm(
            input_size=input_size,
            task_type="classification",
            model_size="medium"
        )
        
        trainer = StockLSTMTrainer(model, model.config)
        optimizer = trainer.configure_optimizer(learning_rate)
    
    scheduler = trainer.configure_scheduler(optimizer, epochs, steps_per_epoch=len(train_loader))
    criterion = nn.BCELoss() 
    
    logger.info(f"Starting training for {epochs} epochs...")
    model.train()
    
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(data)
            predictions = outputs['output'].squeeze()
            
            loss = criterion(predictions, target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred_classes = (predictions > 0.5).float()
            accuracy = (pred_classes == target.float()).float().mean()
            
            train_loss += loss.item()
            train_acc += accuracy.item()
        
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                predictions = outputs['output'].squeeze()
                
                loss = criterion(predictions, target.float())
                pred_classes = (predictions > 0.5).float()
                accuracy = (pred_classes == target.float()).float().mean()
                
                val_loss += loss.item()
                val_acc += accuracy.item()
        
        model.train()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f'Epoch {epoch:3d}/{epochs}: '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': model.config,
                'feature_names': train_ds.feature_names,
                'scaler': train_ds.get_scaler()
            }, model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no val loss improvement in {patience} epochs)")
                break
    logger.info(f"Training completed! Best model saved to {model_save_path}")
    return model

def incremental_train(csv_path="temp.csv",  model_path="stock_model.pth", epochs=10, learning_rate=5e-4):
    """
    Incremental training - continues training existing model with new data
    
    Args:
        csv_path: Path to updated CSV file
        model_path: Path to existing model
        epochs: Number of additional epochs
        learning_rate: Lower learning rate for fine-tuning
    
    Returns:
        Updated model
    """
    return train_stock_model(
        csv_path=csv_path,
        epochs=epochs,
        learning_rate=learning_rate,
        model_save_path=model_path,
        incremental=True,
        existing_model_path=model_path
    )

def train_on_new_data_only(csv_path="temp.csv",model_path="stock_model.pth", days_back=30,epochs=5):
    """
    Train only on recent data for quick daily updates
    
    Args:
        csv_path: Path to CSV file
        model_path: Path to existing model
        days_back: How many recent days to train on
        epochs: Number of epochs (keep small for quick updates)
    
    Returns:
        Updated model
    """
    
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    recent_df = df.tail(days_back + 60).reset_index(drop=True)  # +60 for window_size
    
    temp_path = "temp_recent.csv"
    recent_df.to_csv(temp_path, index=False)
    
    model = train_stock_model(
        csv_path=temp_path,
        epochs=epochs,
        learning_rate=1e-4,  # Lower learning rate
        model_save_path=model_path,
        incremental=True,
        existing_model_path=model_path
    )
    
    import os
    os.remove(temp_path)
    
    return model

def load_trained_model(model_path="stock_model.pth"):
    """
    Load a previously trained model
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model and associated data
    """
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    
    model = create_stock_lstm(
        input_size=checkpoint['config'].input_size,
        task_type=checkpoint['config'].task_type,
        model_size="medium"
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'model': model,
        'scaler': checkpoint['scaler'],
        'feature_names': checkpoint['feature_names'],
        'config': checkpoint['config']
    }

def predict_next_day(csv_path="temp.csv", model_path="stock_model.pth", minutes_ahead=10):
    """
    Make prediction for next day using trained model
    
    Args:
        csv_path: Path to your CSV file
        model_path: Path to trained model
    
    Returns:
        Prediction probability
    """
    model_data = load_trained_model(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    
    config = DatasetConfig(window_size=60, horizon=minutes_ahead, target_type="classification")
    train_ds, _, _ = create_stock_datasets(csv_path, config)
    
    last_sequence = train_ds.X[-1].unsqueeze(0)  # Add batch dimension
    
    model.eval()
    with torch.no_grad():
        outputs = model(last_sequence)
        probability = outputs['output'].squeeze().item()
    
    return probability

if __name__ == "__main__":
    
    # print("=== Full Retraining ===")
    # model = train_stock_model("temp.csv", epochs=100)
    
    # print("\n=== Incremental Training ===")
    # model = train_stock_model("temp.csv", epochs=50, model_save_path="incremental_model.pth")
    # model = incremental_train("temp.csv", "incremental_model.pth", epochs=10)
    
    # print("\n=== Recent Data Training ===")
    # model = train_on_new_data_only("temp.csv", "incremental_model.pth", days_back=30, epochs=5)
    
    # prediction = predict_next_day("temp.csv", "incremental_model.pth")
    # print(f"\nProbability of price going up tomorrow: {prediction:.4f}")
    # print(f"Prediction: {'UP' if prediction > 0.5 else 'DOWN'}")
    
    print("\n=== Daily Usage Recommendations ===")
    print("For daily updates, choose one:")
    print("1. Full retrain (1-2x per week): train_stock_model('temp.csv', epochs=50)")
    print("2. Incremental (daily): incremental_train('temp.csv', 'model.pth', epochs=10)")  
    print("3. Recent data only (daily, fastest): train_on_new_data_only('temp.csv', 'model.pth')")