import numpy as np
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

layer_count = 3

def project(cls1_data, cls2_data, focus, model, language, projection='mds', setsize=None, with_debiasing=None, figname=None):
    if type(cls1_data) == list:
        n_layers = cls1_data[0].shape[1]

        #fig, axes = plt.subplots(3, 4, figsize=(9, 12))

        if n_layers == 12:
            LAYERS = [1-1, 6-1, 12-1][-layer_count:]
            #ymax = 20
            
            
        elif n_layers == 6:
            LAYERS = [1-1, 3-1, 6-1][-layer_count:]
            #ymax = 5
            
        fig, axes = plt.subplots(1, layer_count, figsize=(layer_count*3, 3))

        if model == 'BERT':
            model_alias = 'BERT'
        if model == 'MT' and language == 'DE':
            model_alias = 'MT (EN>DE)'
        if model == 'MT' and language == 'DE-EL':
            model_alias = 'MT (EN>DE+EL)'                    

        #fig.suptitle(f'{model_alias} representations', fontsize=12)

        fig.tight_layout(pad=0.1)
        #flataxes = [ax for tmp in axes for ax in tmp]
        flataxes = axes

        setsize = len(cls1_data)
        #if setsize > 500: setsize = 500

        X = np.r_[np.concatenate(cls1_data, axis=0)[:setsize,:,:], np.concatenate(cls2_data, axis=0)[:setsize,:,:]]
            
        if with_debiasing:
            X = with_debiasing.clean_data(X)

        for l, layer in enumerate(LAYERS):
            print('plotting layer %d' % (layer+1))            

            if projection == 'mds':
                mds = MDS(n_components=2)
                X_transformed = mds.fit_transform(X[:,layer,:].astype(np.float64))
            if projection == 'pca':
                pca = PCA(n_components=2)
                X_transformed = pca.fit_transform(X[:,layer,:].astype(np.float64))
            if projection == 'tsne':
                tsne = TSNE(n_components=2, verbose=1)
                X_transformed = tsne.fit_transform(X[:,layer,:].astype(np.float64))

            colors = ['red']*setsize + ['blue']*setsize

            if layer_count == 1:
                ax = flataxes
            elif layer_count == 3:
                ax = flataxes[l]

            ax.set_aspect('equal', adjustable='box')
            ax.scatter(X_transformed[:,0], X_transformed[:,1], s=2, c=colors)

            #ax.set_xlim((-20, 20))
            #ax.set_ylim((-20, 20))

            focus_aliases = {'verb': 'VERBS', 'subject': 'SUBJECTS', 'object':'OBJECTS'}
            
            if layer_count == 1:
                ax.set_title(f'{focus_aliases[focus]}', fontsize=24)
            elif layer_count == 3:
                ax.set_title(f'Layer {LAYERS[l]+1}', fontsize=24)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            print('plotting done.')

    else:
        if not setsize:
            setsize = cls1_data.shape[0]
        
        n_layers = cls1_data.shape[1]

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


