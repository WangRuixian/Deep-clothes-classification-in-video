import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()# 函数用于创建子图
people = ("trousers", "short_sleeved_shirt", "shorts", "skirt",
          "long_sleeved_shirt", "long_sleeved_outwear","vest","vest_dress","short_sleeved_dress",
          "sling_dress","long_sleeved_dress","sling","short_sleeved_outwear")
y_pos = np.arange(len(people))
performance = (0.94,0.91,0.86,0.81,0.80,0.78,0.76,0.75,0.71,0.67,0.46,0.41,0.37)
ax.barh(y_pos, performance, color = "#6699cc")
list=np.array([0.94,0.91,0.86,0.81,0.80,0.78,0.76,0.75,0.71,0.67,0.46,0.41,0.37])
for a, b in zip(performance,y_pos):
	plt.text(a+0.025, b + 0.1, '%.2f' % performance[b], ha='center', va='bottom', fontsize=7)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()
ax.set_xlabel("Average Precision",loc='center', fontsize='15')
ax.set_title("mAP= 71.05%",loc='center', fontsize='15',
        )
plt.show()
