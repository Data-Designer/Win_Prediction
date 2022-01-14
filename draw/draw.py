import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

y = np.random.uniform(0.4,0.65,(30,)).tolist()
y2 = np.random.uniform(0.60,0.75,(45,)).tolist()
y3 = np.random.uniform(0.7,0.85,(30,)).tolist()
y.extend(y2)
y.extend(y3)
z = 1 - np.array(y)
x = list(range(len(y)))
plt.plot(x, y, '--', color='b', linewidth=1.5, marker='^',label='Team1')
plt.plot(x, z, '--', color='r', linewidth=1.5, marker='^',label='Team2')
plt.xlabel(u'Time/s', fontdict={'family': 'Times New Roman',
                                                 'color': 'black',
                                                 'weight': 'normal',
                                                 'size': 13})
plt.ylabel(u'Win Rate', fontdict={'family': 'Times New Roman',
                                                  'fontstyle': 'italic',
                                                  'color': 'black',
                                                  'weight': 'normal',
                                                  'size': 13})
plt.legend()
plt.show()