import matplotlib.pyplot as plt
import numpy as np

# Data from table_performance_comparison.tex (Post Content Gain, assuming this is "modality gains")
algorithms = ['MBPO', 'PPO']
means = [0.035, 0.047]
stds = [0.009, 0.001]

# Create bar plot with error bars
fig, ax = plt.subplots()
ax.bar(algorithms, means, yerr=stds, capsize=5, color=['blue', 'green'])
ax.set_ylabel('Post Content Gain (Modality Gains)')
ax.set_title('Comparison of Modality Gains: MBPO vs PPO')
ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig('modality_gains_mbpo_ppo.png')
plt.show()