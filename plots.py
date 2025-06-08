#plots
import matplotlib.pyplot as plt

# Data
epochs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
easy =   [1.3891, 3.4927, 6.4127, 15.2264, 18.7644, 25.6150, 26.7405, 35.8973, 43.2299, 44.5223]
medium = [1.9054, 3.9568, 7.4925, 14.2649, 15.6451, 23.1944, 22.6126, 32.1691, 37.3545, 38.2343]
hard =   [1.9284, 4.2384, 8.0846, 13.0553, 14.8211, 19.9132, 19.6711, 28.2433, 32.8130, 34.5861]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, easy, label='Easy', marker='o')
plt.plot(epochs, medium, label='Medium', marker='s')
plt.plot(epochs, hard, label='Hard', marker='^')

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Performance (%)')
plt.title('Model Performance Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('performance_plot.png', dpi=300)
# Show the plot
plt.show()