import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_context("paper")
import scipy.stats as st

class Recorder():
    def __init__(self, n_removes, algos):
        self.n_removes = n_removes
        self.algos = algos
        self.test_D = {}
        self.auc_D = {}
        for algo in algos:
            self.test_D[algo] = {p:[] for p in self.n_removes}
            self.auc_D[algo] = []
   
    def update(self, D):
        for algo in self.algos:
            aucs = np.zeros(len(D[algo][0]))
            for i, p in enumerate(self.n_removes):
                self.test_D[algo][p] += D[algo][i]
                if i == 0:
                    denom = self.n_removes[i]
                else:
                    denom = self.n_removes[i] - self.n_removes[i-1]
                aucs += np.array(D[algo][i]) * denom / self.n_removes[-1] 
                 
            self.auc_D[algo] += list(aucs)
                
    def get_separate_mean_and_conf_interval(self, D):
        mean_D, conf_interval_D = {}, {}
        for algo in self.algos:
            mean_D[algo] = []
            conf_interval_D[algo] = [] 
            for p in self.n_removes:
                 mean_D[algo].append(np.mean(D[algo][p]) )
                 a, b = st.norm.interval(alpha=0.95, loc=np.mean(D[algo][p]), scale=st.sem(D[algo][p]))
                 conf_interval_D[algo].append((b-a)/2)
        
        return mean_D, conf_interval_D
    
    def get_auc_mean_and_conf_interval(self, D):
        mean_D, conf_interval_D = {}, {} 
        for algo in self.algos:
            mean_D[algo] = np.mean(self.auc_D[algo]) 
            a, b = st.norm.interval(alpha=0.95, loc=np.mean(self.auc_D[algo]), scale=st.sem(self.auc_D[algo]))
            #conf_interval_D[algo] = np.std(self.auc_D[algo])
            conf_interval_D[algo] = (b-a) / 2
        
        return mean_D, conf_interval_D
    
    def get_auc_scores(self):
        mean_auc, conf_interval_auc = self.get_auc_mean_and_conf_interval(self.test_D)
        return mean_auc, conf_interval_auc
     
    def print(self, file_name = None):
        mean_test, conf_interval_test = self.get_separate_mean_and_conf_interval(self.test_D)
        mean_auc, conf_interval_auc = self.get_auc_mean_and_conf_interval(self.test_D)
        
        f = None
        if file_name is not None:
            f = open(file_name, "a")
        
        print("Deletion curves:", file = f)
        s = "{:40s} " + "{:22s} " * (len(self.n_removes) + 1)
        L = ["Number of training data dropped"] + [ str(x) for x in self.n_removes] + ["AUC"]
        #L = ["Percentage of negative samples removed"] + [ str(x) for x in self.n_removes] + ["AUC"]
        print(s.format(*L), file = f)
        for algo in self.algos:
            s = "{:40s} " + "{:10s}+-{:10s} " * (len(self.n_removes) + 1 )
            L = [algo] 
            for i in range(len(self.n_removes)):
                L.append("%0.3f" % mean_test[algo][i])
                L.append("%0.3f" % conf_interval_test[algo][i])
            L.append("%0.3f" % mean_auc[algo])
            L.append("%0.3f" % conf_interval_auc[algo])
            print(s.format( *L), file = f)
    
    def change_algo_order(self, algo):
        self.algos = algo
            
    def plot(self, x_axis_name, y_axis_name, save_fig_path, show_conf_interval = False):
        mean_test, conf_interval_test = self.get_separate_mean_and_conf_interval(self.test_D)
        sns.set_theme()
        data = { "Algo":[], x_axis_name:[], y_axis_name:[]}
        for algo in self.algos:
            for i, n in enumerate(self.n_removes):
                if i == 4:
                    break
                data['Algo'].append(algo_name_map[algo])
                data[x_axis_name].append(0.1 * i)
                if "found" in y_axis_name:
                    total = int(55375/2)
                    data[y_axis_name].append(mean_test[algo][i] / total * 100 )
                else:
                    data[y_axis_name].append(mean_test[algo][i])
                #data[y_axis_name].append(0.1 * i)

        data = pd.DataFrame.from_dict(data)
        
        f, axes = plt.subplots(1, 1,  figsize=(4.6, 4.6))            

        palette = sns.color_palette("colorblind", len(self.algos))
        sns.lineplot(x = x_axis_name, y = y_axis_name, data = data, hue='Algo', 
                     ax = axes, palette= palette)
        handles, labels = axes.get_legend_handles_labels()
        axes.legend(handles=handles, labels=labels) 
        
        if show_conf_interval:
            for algo in self.algos:
                #plt.fill_between(self.n_removes , np.array(mean_test[algo]) - np.array(conf_interval_test[algo]), 
                #                np.array(mean_test[algo]) + np.array(conf_interval_test[algo]), alpha=.3)
                ps = [0, 0.1, 0.2, 0.3]
                plt.fill_between(ps , np.array(mean_test[algo])[:-1] - np.array(conf_interval_test[algo])[:-1], 
                                np.array(mean_test[algo])[:-1] + np.array(conf_interval_test[algo])[:-1], alpha=.3)
        
        plt.tight_layout()
        
        if not save_fig_path.endswith('.png'):
            save_fig_path += '.png'
        plt.savefig(save_fig_path)
        
    def merge(self, recorder):
        assert set(self.n_removes ) == set(recorder.n_removes)
        assert set(self.algos) == set(recorder.algos)
        
        for algo in self.algos:
            aucs = np.zeros(len(recorder.test_D[algo][self.n_removes[0]]))
            for i, p in enumerate(self.n_removes):
                self.test_D[algo][p] += recorder.test_D[algo][p]
                if i == 0:
                    denom = self.n_removes[i]
                else:
                    denom = self.n_removes[i] - self.n_removes[i-1]
                aucs += np.array(recorder.test_D[algo][p]) * denom / self.n_removes[-1] 
                
            self.auc_D[algo] += list(aucs)
    
        