import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_log.csv')

df.plot( x = 'episode', y = 'reward')

plt.savefig('training_log.png')
plt.show()