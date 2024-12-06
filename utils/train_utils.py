import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import os
from model.mnist_model import MNIST_DNN, count_parameters
from utils.logger import setup_logger
from datetime import datetime

def save_model(model, accuracy, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model with timestamp and accuracy
    model_path = f'models/mnist_model_{timestamp}_acc{accuracy:.2f}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'timestamp': timestamp,
        'architecture': str(model),
        'parameters': count_parameters(model)
    }, model_path)
    
    return model_path

def train_model(epochs=1, batch_size=32, learning_rate=0.001):
    # Setup logger
    logger, log_file = setup_logger()
    
    # Create run configuration
    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dropout_rate': 0.1,
        'timestamp': os.path.basename(log_file).split('_')[2].split('.')[0]
    }
    
    # Log configuration
    logger.info("Starting new training run")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Set device
    device = torch.device(config['device'])
    
    # Initialize model
    model = MNIST_DNN()
    total_params = count_parameters(model)
    
    # Log model details
    logger.info(f"\nModel Architecture:\n{str(model)}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Parameter Budget: 25,000")
    logger.info(f"Budget Remaining: {25000 - total_params:,}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    logger.info(f"Training samples: {len(train_dataset):,}")
    logger.info(f"Testing samples: {len(test_dataset):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    best_accuracy = 0
    training_history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                batch_metrics = {
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'accuracy': 100. * correct/total
                }
                logger.info(
                    f"Epoch: {epoch+1}/{epochs} "
                    f"[{batch_idx * len(data):>5d}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)] "
                    f"Loss: {loss.item():.4f} "
                    f"Accuracy: {100. * correct/total:.2f}%"
                )
                training_history.append(batch_metrics)
        
        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        logger.info("Starting evaluation...")
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= total
        accuracy = 100. * correct / total
        best_accuracy = max(best_accuracy, accuracy)
        
        eval_metrics = {
            'epoch': epoch + 1,
            'test_loss': test_loss,
            'test_accuracy': accuracy,
            'best_accuracy': best_accuracy
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Average Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {correct}/{total} ({accuracy:.2f}%)")
        logger.info(f"Best Accuracy: {best_accuracy:.2f}%")
        
        # Save metrics
        metrics_file = f'logs/metrics_{config["timestamp"]}.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'config': config,
                'training_history': training_history,
                'final_metrics': eval_metrics
            }, f, indent=2)
    
    logger.info("\nFinal Results:")
    logger.info(f"Model Parameters: {total_params:,}")
    logger.info(f"Final Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Best Test Accuracy: {best_accuracy:.2f}%")
    logger.info(f"Log file saved: {log_file}")
    logger.info(f"Metrics saved: {metrics_file}")
    
    # After training, save the model
    model_path = save_model(model, best_accuracy, config['timestamp'])
    logger.info(f"Model saved: {model_path}")
    
    return model, best_accuracy, test_loss 