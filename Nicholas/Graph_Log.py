import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_log.csv')

df.plot()

plt.show()