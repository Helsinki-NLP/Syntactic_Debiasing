import os, sys
import numpy as np
import matplotlib.pyplot as plt

import cca_core #https://github.com/google/svcca
import pwcca


class CCA:
    def __init__(self, group_1_cls1_list, group_1_cls2_list, group_1_layer, 
                       group_2_cls1_list, group_2_cls2_list, group_2_layer):

        group_1_cls1_np = np.concatenate(group_1_cls1_list, axis=0)
        group_1_cls2_np = np.concatenate(group_1_cls2_list, axis=0)
        group_2_cls1_np = np.concatenate(group_2_cls1_list, axis=0)
        group_2_cls2_np = np.concatenate(group_2_cls2_list, axis=0)

        len_g1_c1 = group_1_cls1_np.shape[0]
        len_g1_c2 = group_1_cls2_np.shape[0]
        len_g2_c1 = group_2_cls1_np.shape[0]
        len_g2_c2 = group_2_cls2_np.shape[0]

        min_len_c1 = min(len_g1_c1, len_g2_c1)
        min_len_c2 = min(len_g1_c2, len_g2_c2)

        group_1_cls1_np = group_1_cls1_np[:min_len_c1, group_1_layer, :]
        group_1_cls2_np = group_1_cls2_np[:min_len_c2, group_1_layer, :]

        group_2_cls1_np = group_2_cls1_np[:min_len_c1, group_2_layer, :]
        group_2_cls2_np = group_2_cls2_np[:min_len_c2, group_2_layer, :]

        self.A = np.concatenate([group_1_cls1_np, group_1_cls2_np], axis=0)
        self.B = np.concatenate([group_2_cls1_np, group_2_cls2_np], axis=0)

        self.A = self.normalize(self.A)
        self.B = self.normalize(self.B)

        print(self.A.shape)
        print(self.B.shape)


    def normalize(self, a):
        row_sums = a.sum(axis=1)
        return a / row_sums[:, np.newaxis]


    def _plot_helper(self, arr, xlabel, ylabel):
        plt.plot(arr, lw=2.0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()


    def apply_svcca(self):
        results = cca_core.get_cca_similarity(self.A, self.B, verbose=True, epsilon=1e-5)
        print('return_dict["coef_x"]:', results["coef_x"])
        print('return_dict["coef_y"]:', results["coef_y"])

        print(results["coef_x"].shape)
        print(results["cca_coef1"].shape)

        plt.figure()
        plt.imshow(results["coef_x"])

        plt.figure()
        plt.imshow(results["coef_y"])

        self._plot_helper(results["cca_coef1"], "CCA coef idx", "CCA coef value")


    def apply_pwcca(self):
        pwcca_mean, w, _ = pwcca.compute_pwcca(self.A, self.B, epsilon=1e-5)

        print("Results using PWCCA: ", pwcca_mean)

