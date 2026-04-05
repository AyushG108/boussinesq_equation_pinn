x_values = np.linspace(-20, 20, 100)  # Adjust the number of points as needed
t_values = np.linspace(-5, 5, 100)   # Adjust the number of points as needed

# Create a meshgrid for x and t values
x_grid, t_grid = np.meshgrid(x_values, t_values)
x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)
t_tensor = torch.tensor(t_grid, dtype=torch.float32).view(-1, 1)

# Concatenate x and t tensors to create input tensor for prediction
input_tensor = torch.cat((x_tensor, t_tensor), dim=1)

# Get predictions for the entire grid
with torch.no_grad():
    y_pred = model(input_tensor)
    y_pred = y_pred.cpu().numpy().reshape(len(t_values), len(x_values))

# Create a continuous heatmap
plt.figure(figsize=(6, 10))
sns.heatmap(y_pred, cmap="jet", xticklabels=10, yticklabels=10)
plt.title('Continuous Heatmap of the Predicted Function')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()