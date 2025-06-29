import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_log_test.csv')
df.plot( x = 'episode', y = 'reward')
plt.savefig('training_log.png')

ef = pd.read_csv('eval_log.csv')

ef.plot( x = 'episode', y = 'avg_reward')

plt.savefig('eval_log_125000.png')
plt.show()
