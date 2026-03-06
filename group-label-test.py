import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns


def spread_statistic(x):
	return max(x) - min(x) - (len(x) - 1)


def spread_test(ordered, labels, B):
	"""Run the group label hypothesis test.

	Parameters
	----------
	ordered: np.ndarray
		Sorted values of removal times
	labels: np.ndarray
		Group labels
	B: int
		Number of shufflings/permutations

	Returns
	-------
	dict
	"""
	statistics = np.zeros(B)
	uniq_labels = np.unique(labels)

	statistic = 0
	for u in uniq_labels:
		x = ordered[labels == u]
		statistic += spread_statistic(x)
	test_value = deepcopy(statistic)

	labels_permuted = labels.copy()
	for b in range(B):
		statistic = 0
		np.random.shuffle(labels_permuted)
		for u in uniq_labels:
			x = ordered[labels_permuted == u]
			statistic += spread_statistic(x)
		statistics[b] = statistic

	p = sum(test_value > statistics) / B

	return {
		"p_value": p,
		"test_statistic": test_value,
		"distribution": statistics,
	}


table = pd.read_csv("measles_hagelloch.csv", sep=",")
i = table["numeric_prodrome"]
l = table["class"]

print(l.value_counts())

sns.stripplot(
	table,
	x="class",
	y="numeric_prodrome",
	hue="class",
	palette="tab10",
)
plt.legend(title="Class")
plt.ylabel("Time of prodromal symptoms")
plt.xlabel("School class")

l = l[(i > 20) & (i < 50)]
i = i[(i > 20) & (i < 50)]

i = i.to_numpy()
l = l.to_numpy()
sorting = np.argsort(i)
print(sorting[:5])

output = spread_test(sorting, l, 100000)
print(output)

plt.figure()
plt.hist(output["distribution"], 30, edgecolor="k")
plt.axvline(output["test_statistic"], color="tab:orange", linewidth=3, linestyle="--")
plt.title(f"p-value is {output['p_value']}")
plt.xlabel("Test statistic")
plt.yticks([])
plt.xlim(280, 350)
plt.show()
