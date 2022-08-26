from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def remove_outliers(data, r=2.0):
    outliers_data = abs(data - np.mean(data, axis=0)) >= r * np.std(data, axis=0)
    outliers = np.any(outliers_data, axis=1)
    keep = np.logical_not(outliers)
    return outliers, keep
f = open('data/labels.txt', 'r', encoding='utf-8')
labels = []
for line in f.readlines():
    line = line.strip()
    labels.append(int(line))

x = np.load("/common-data/new_build/xingrui.lou/Dul_attention_cvae/M2.npy")
print(x.shape)
X_emb_2d = TSNE(n_components=2, verbose=1, perplexity=40).fit_transform(x)


outliers, keep = remove_outliers(X_emb_2d)

X_emb_2d = X_emb_2d[keep, :]
y = [l for l, k in zip(labels, keep.tolist()) if k]

# plot
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0, 0, 1, 1])
cc = ['r', 'b', 'g', 'y', 'k', 'c'] # other:red like:blue sadness:green disgust:yellow anger:black happiness:cyan
for i, l in enumerate(sorted(set(y))):
    idx = [yl == l for yl in y]
    plt.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1], c=cc[i], s=10, edgecolor='none', alpha=0.5)
ax.axis('off')  # adding it will get no axis
plt.savefig('latent_e.png')
plt.show()
