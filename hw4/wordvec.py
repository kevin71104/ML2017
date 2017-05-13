import word2vec as w2v
import numpy as np
from sklearn.manifold import TSNE
import nltk
import matplotlib.pyplot as plt
from adjustText import adjust_text

w2v.word2phrase('all.txt','all-phrases.txt',verbose = True)
w2v.word2vec(train='all-phrases.txt',
             output='all.bin',
             cbow=1, # 1 :skip-gram
             size=200,
             window=5,
             negative=10,
             min_count=10,
             alpha=0.05,
             verbose=True
            )

model = w2v.load('all.bin')

plot_num = 800
vocabs = []
vecs = []
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vecs = np.array(vecs)[:plot_num]
vocabs = vocabs[:plot_num]

# compress to 2 dimensionality
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)

# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]

plt.figure(figsize=(20,10))

texts = []
for i, label in enumerate(vocabs): # i is the index, label is the vocab
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

plt.savefig('./figure/hp.png', dpi=600)
#plt.show()
