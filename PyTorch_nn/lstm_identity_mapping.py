import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
sequence_length = 10
num_samples = 1000
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Data range
data_min = -1.3
data_max = 4.0

# Generate random data in [-1.3, 4.0]
def generate_data(num_samples, seq_length, data_min, data_max):
    """Generate random sequences where output should equal input"""
    data = np.random.uniform(data_min, data_max, (num_samples, seq_length, 1))
    # For identity mapping, target is the same as input
    return data, data.copy()

# Generate training and test data
train_data, train_targets = generate_data(num_samples, sequence_length, data_min, data_max)
test_data, test_targets = generate_data(200, sequence_length, data_min, data_max)

# Convert to PyTorch tensors
train_data = torch.FloatTensor(train_data)
train_targets = torch.FloatTensor(train_targets)
test_data = torch.FloatTensor(test_data)
test_targets = torch.FloatTensor(test_targets)

# Create DataLoader
train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define LSTM Model
class LSTMIdentity(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMIdentity, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of each time step
        out = self.fc(out)
        return out

# Initialize model
model = LSTMIdentity(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Training LSTM for Identity Mapping...")
print(f"Data range: [{data_min}, {data_max}]")
print(f"Sequence length: {sequence_length}")
print(f"Training samples: {num_samples}")
print("-" * 50)

train_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_data, batch_targets in train_loader:
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

print("\nTraining completed!")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_predictions = model(test_data)
    test_loss = criterion(test_predictions, test_targets)
    print(f'\nTest Loss: {test_loss.item():.6f}')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training Loss
axes[0, 0].plot(train_losses, linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('MSE Loss', fontsize=12)
axes[0, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Predicted vs Real (scatter plot for all timesteps)
test_pred_flat = test_predictions.numpy().flatten()
test_real_flat = test_targets.numpy().flatten()

axes[0, 1].scatter(test_real_flat, test_pred_flat, alpha=0.5, s=10)
axes[0, 1].plot([data_min, data_max], [data_min, data_max], 'r--', linewidth=2, label='Perfect Identity')
axes[0, 1].set_xlabel('Real Values', fontsize=12)
axes[0, 1].set_ylabel('Predicted Values', fontsize=12)
axes[0, 1].set_title('Predicted vs Real (All Test Points)', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axis('equal')

# Calculate R² score
correlation_matrix = np.corrcoef(test_real_flat, test_pred_flat)
r_squared = correlation_matrix[0, 1] ** 2
axes[0, 1].text(0.05, 0.95, f'R² = {r_squared:.4f}', 
                transform=axes[0, 1].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Sample sequences comparison
num_samples_to_plot = 5
for i in range(num_samples_to_plot):
    axes[1, 0].plot(test_targets[i].numpy().flatten(), 
                    label=f'Real {i+1}', linestyle='-', marker='o', markersize=4)
    axes[1, 0].plot(test_predictions[i].numpy().flatten(), 
                    label=f'Pred {i+1}', linestyle='--', marker='x', markersize=4)

axes[1, 0].set_xlabel('Time Step', fontsize=12)
axes[1, 0].set_ylabel('Value', fontsize=12)
axes[1, 0].set_title('Sample Sequences: Real vs Predicted', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=8, ncol=2)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Error distribution
errors = test_pred_flat - test_real_flat
axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
axes[1, 1].set_xlabel('Prediction Error', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

# Add statistics
mean_error = np.mean(errors)
std_error = np.std(errors)
axes[1, 1].text(0.05, 0.95, f'Mean Error: {mean_error:.4f}\nStd Error: {std_error:.4f}', 
                transform=axes[1, 1].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('./lstm_identity_mapping_results.png', dpi=300, bbox_inches='tight')
print("\nFigure saved to 'lstm_identity_mapping_results.png'")

# Print final statistics
print("\n" + "="*50)
print("Final Results:")
print("="*50)
print(f"Mean Absolute Error: {np.mean(np.abs(errors)):.6f}")
print(f"Root Mean Squared Error: {np.sqrt(np.mean(errors**2)):.6f}")
print(f"R² Score: {r_squared:.6f}")
print(f"Correlation: {correlation_matrix[0, 1]:.6f}")
print("="*50)
print("\nConclusion: LSTM", "CAN" if r_squared > 0.95 else "has difficulty to", 
      "learn identity mapping with R² =", f"{r_squared:.4f}")
