import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_log.csv')
df.plot( x = 'episode', y = 'reward')
plt.savefig('training_log.png')

ef = pd.read_csv('eval_log.csv')

ef.plot( x = 'episode', y = 'avg_reward')

plt.savefig('eval_log.png')
plt.show()