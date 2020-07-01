import numpy as np
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def plot_mds(cls1_data, cls2_data, setsize=None, with_debiasing=None, figname=None):
    if type(cls1_data) == list:
        fig, axes = plt.subplots(3, 4, figsize=(9, 12))

        fig.tight_layout(pad=0.5)
        flataxes = [ax for tmp in axes for ax in tmp]

        setsize = len(cls1_data)
        if setsize > 500: setsize = 500

        for layer in range(0, 12):
            print('plotting layer %d' % (layer+1))

            X = np.r_[np.concatenate(cls1_data, axis=0)[:setsize,layer,:], np.concatenate(cls2_data, axis=0)[:setsize,layer,:]]
            
            if with_debiasing:
                X = with_debiasing.clean_data(X)

            mds = MDS(n_components=2)
            X_transformed = mds.fit_transform(X)
            
            colors = ['red']*setsize + ['blue']*setsize

            ax = flataxes[layer]
            ax.set_aspect('equal', adjustable='box')
            ax.scatter(X_transformed[:,0], X_transformed[:,1], s=2, c=colors)

            ax.set_xlim((-20, 20))
            ax.set_ylim((-20, 20))
            ax.set_title('Layer %d' % (layer+1), fontsize=12)

            print('plotting done.')

    else:
        if not setsize:
            setsize = cls1_data.shape[0]

        colors = ['red']*setsize + ['blue']*setsize

        X = np.r_[cls1_data[:setsize,:], cls2_data[:setsize,:]]

        mds = MDS(n_components=2)
        X_transformed = mds.fit_transform(X)

        fig = plt.plot()
        plt.scatter(X_transformed[:,0], X_transformed[:,1], s=2, c=colors)

    if figname:
        plt.savefig(figname)
    else:
        plt.show()


def plot_pca(cls1_data, cls2_data, setsize=None, figname=None):
    if not setsize:
        setsize = cls1_data.shape[0]

    colors = ['red']*setsize + ['blue']*setsize

    X = np.r_[cls1_data[:setsize,:], cls2_data[:setsize,:]]

    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    fig = plt.plot()
    plt.scatter(X_transformed[:,0], X_transformed[:,1], s=2, c=colors)

    if figname:
        plt.savefig(figname)
    else:
        plt.show()