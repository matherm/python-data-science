import pickle
import numpy as np
from ummon import *
import torch
import matplotlib.pyplot as plt
from ummon.predictor import Predictor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import sklearn.svm as svm
import numpy as np

def save_obj(obj, name ):
    with open('local/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('local/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def flatten_transform(tensor):
    return tensor.contiguous().view(-1).clone()   


def imshow(X):
     X = X.reshape((3, 28,28))
     X = np.transpose(X,(1,2,0))
     plt.imshow(X/255)
     plt.show()


class Evaluator():
    
    def __init__(self, model, data_train, data_val, data_novelty):
        self.model = model
        self.data_train = data_train
        self.data_val = data_val
        self.data_novelty = data_novelty
        

    def predict_sample_data(self):
        model, data_train, data_val, data_novelty = self.model, self.data_train, self.data_val, self.data_novelty
        
        
        noise_u = np.random.uniform(0,1,data_val.shape[1]*data_val.shape[0]).reshape(-1, data_val.shape[1]).astype(np.float32)
        noise_w = np.random.normal(0,1,data_val.shape[1]*data_val.shape[0]).reshape(-1, data_val.shape[1]).astype(np.float32)
    
        noise_u_t = Predictor.predict(model, noise_u)
        noise_w_t = Predictor.predict(model, noise_w)
        val_t = Predictor.predict(model, data_val)
        val_o = model.o.data.cpu().numpy()
        novel_t = Predictor.predict(model, data_novelty)
        novel_o = model.o.data.cpu().numpy()
        train_t = Predictor.predict(model, data_train)
        train_o = model.o.data.cpu().numpy()
        return noise_u_t, noise_w_t, val_t, val_o, novel_t, novel_o, train_t, train_o
    
    
    def evaluate_model(self, *args):
        model, data_train, data_val, data_novelty = self.model, self.data_train, self.data_val, self.data_novelty
        noise_u_t, noise_w_t, val_t, val_o, novel_t, novel_o, train_t, train_o = self.predict_sample_data()
        
        if hasattr(args[0], "plots") and args[0].plots == True:
            import matplotlib.pyplot as plt
            clip = np.inf
            plt.hist(np.clip(np.sum(train_t**2, 1), 0, clip), color="b", alpha=0.5, bins=50)
            plt.hist(np.clip(np.sum(noise_u_t**2, 1), 0, clip), color="g", alpha=0.5, bins=50)
            plt.hist(np.clip(np.sum(noise_w_t**2, 1), 0, clip), color="y", alpha=0.5, bins=50)
            plt.hist(np.clip(np.sum(val_t**2, 1), 0, clip), alpha=0.5, bins=50)
            plt.hist(np.clip(np.sum(novel_t**2, 1), 0, clip), color="r", alpha=0.5, bins=50)
            plt.title("Distance to origin (euclidean) ")
            plt.show()
            plt.hist(np.clip(np.sum((train_o - data_train)**2, 1), 0, clip), color="b", alpha=0.5, bins=50)
            plt.hist(np.clip(np.sum((val_o - data_val)**2, 1), 0, clip), alpha=0.5, bins=50)
            plt.hist(np.clip(np.sum((novel_o -data_novelty)**2, 1), 0, clip), color="r", alpha=0.5, bins=50) 
            plt.title("MSE (reconstruction loss)")
            plt.show()
            plt.hist(np.sum(noise_w_t, 1), color="y", alpha=0.5, bins=50)
            plt.hist(np.sum(val_t, 1), alpha=0.5, bins=50)
            plt.hist(np.sum(novel_t, 1), color="r", alpha=0.5, bins=50)
            plt.hist(np.sum(train_t, 1), color="b", alpha=0.5, bins=50)
            plt.hist(np.sum(noise_u_t, 1), color="g", alpha=0.5, bins=50)
            plt.title("Sum of means (isotropic gaussian)")
            plt.show()
          
            try:
                rect = int(data_val.shape[1]**0.5)
                plt.imshow(model.o[10].data.cpu().numpy().reshape(rect,rect))
                plt.title("Reconstruction (good)")
                plt.show()
                plt.imshow(model.affine_dec(torch.from_numpy(np.random.uniform(-10,10, noise_u_t.shape[1])).float()).data.numpy().reshape(rect,rect))
                plt.title("Reconstruction (unform random)")
                plt.show()
                plt.imshow(novel_o[0].reshape(rect,rect))
                plt.title("Reconstruction (novelty - zero)")
                plt.show()
                plt.imshow(novel_o[10].reshape(rect,rect))
                plt.title("Reconstruction (novelty - nine)")
                plt.show()
                plt.imshow(novel_o[6].reshape(rect,rect))
                plt.title("Reconstruction (novelty - seven)")
                plt.show()
            except:
                pass
            
        from sklearn.metrics import roc_auc_score
        arocs = {}
   #     y_true = np.array([1] * val_t.shape[0] + [-1] * novel_t.shape[0])
   #     y_scores = np.hstack((np.sum(val_t**2, 1), np.sum(novel_t**2, 1)))
   #     print("AUROC LAT (VAL)", roc_auc_score(y_true, y_scores))
   #     arocs["AUROC LAT (VAL)"] = roc_auc_score(y_true, y_scores)
        
   #     y_true = np.array([-1] * val_t.shape[0] + [1] * novel_t.shape[0])
   #     y_scores = np.hstack((np.sum((data_val - val_o)**2, 1), np.sum((novel_o-data_novelty)**2, 1)))
   #     print("AUROC REC (VAL)", roc_auc_score(y_true, y_scores))
   #     arocs["AUROC REC (VAL)"] = roc_auc_score(y_true, y_scores)
        
        
        y_true = np.array([1] * train_t.shape[0] + [-1] * novel_t.shape[0])
        y_scores = np.hstack((np.sum(train_t**2, 1), np.sum(novel_t**2, 1)))
        print("AUROC LAT l2 (TRAIN)", roc_auc_score(y_true, y_scores))
        arocs["AUROC LAT l2 (TRAIN)"] = roc_auc_score(y_true, y_scores)
        y_true = np.array([0] * train_t.shape[0] + [1] * novel_t.shape[0])
        y_scores = np.hstack((percentile_decision_function(np.sum(train_t**2, 1), np.sum(train_t**2, 1)), percentile_decision_function(np.sum(train_t**2, 1), np.sum(novel_t**2, 1))))
        print("AUROC LAT l2 (TRAIN Df)", roc_auc_score(y_true, y_scores))
        arocs["AUROC LAT l2 (TRAIN Df)"] = roc_auc_score(y_true, y_scores)
        
        y_true = np.array([1] * train_t.shape[0] + [-1] * novel_t.shape[0])
        y_scores = np.hstack((np.sum(train_t, 1), np.sum(novel_t, 1)))
        print("AUROC LAT l1 (TRAIN)", roc_auc_score(y_true, y_scores))
        arocs["AUROC LAT l1 (TRAIN)"] = roc_auc_score(y_true, y_scores)
        y_true = np.array([0] * train_t.shape[0] + [1] * novel_t.shape[0])
        y_scores = np.hstack((percentile_decision_function(np.sum(train_t, 1), np.sum(train_t, 1)), percentile_decision_function(np.sum(train_t, 1), np.sum(novel_t, 1))))
        print("AUROC LAT l1 (TRAIN Df)", roc_auc_score(y_true, y_scores))
        arocs["AUROC LAT l1 (TRAIN Df)"] = roc_auc_score(y_true, y_scores)
        
        y_true = np.array([-1] * train_t.shape[0] + [1] * novel_t.shape[0])
        y_scores = np.hstack((np.sum((data_train - train_o)**2, 1), np.sum((novel_o-data_novelty)**2, 1)))
        print("AUROC REC (TRAIN)", roc_auc_score(y_true, y_scores))
        arocs["AUROC REC (TRAIN)"] = roc_auc_score(y_true, y_scores)
        y_true = np.array([0] * train_t.shape[0] + [1] * novel_t.shape[0])
        y_scores = np.hstack((percentile_decision_function(np.sum((data_train - train_o)**2, 1), np.sum((data_train - train_o)**2, 1), anomaly_location='right'), percentile_decision_function(np.sum((data_train - train_o)**2, 1), np.sum((novel_o-data_novelty)**2, 1), anomaly_location='right')))
        print("AUROC REC (TRAIN Df)", roc_auc_score(y_true, y_scores))
        arocs["AUROC REC (TRAIN Df)"] = roc_auc_score(y_true, y_scores)
        return arocs

        
def compareOneClassSVM(nu, data_train_ref, data_novelty, kernel='linear', config_name=None, print=print, plot=True):
    if plot: plt.figure()

    clf = svm.OneClassSVM(nu=nu, kernel=kernel)
    clf.fit(data_train_ref)
    results_train = clf.predict(data_train_ref)
    results_novelty = clf.predict(data_novelty)

    # data_novelty
    if plot: plt.hist((results_novelty), alpha=0.5, bins=50, label='data_novelty')

    # Train
    if plot: plt.hist((results_train), alpha=0.5, bins=50, label='data_train')
    if plot: plt.legend()

    print('OneClassSVM ({}) prediction data_train {}'.format(kernel,
                                                             100 * np.count_nonzero(
                                                                 np.asarray(results_train) == 1) / len(results_train)))

    print('OneClassSVM ({}) prediction data_novelty: {}'.format(kernel,
                                                                100 * np.count_nonzero(
                                                                    np.asarray(results_novelty) == -1) / len(
                                                                    results_novelty)))
    if config_name:
        if plot: plt.title(config_name)
        path = './data/exp_output/' + config_name + '_' + kernel + '.png'
        if plot: plt.savefig(path, format='png')

    # Ref. class data -> label 1
    data_train_y = [1 for i in range(len(results_train))]

    # Novelty data -> label -1
    data_test_y = [-1 for i in range(len(results_novelty))]

    y_true = np.concatenate((data_train_y, data_test_y))
    y_scores = np.concatenate((results_train, results_novelty))
    print('ROC AUC ({}; kernel: {}), computed from prediction scores: {}'.format(config_name, kernel,
                                                                                 roc_auc_score(y_true, y_scores)))
    print('ROC APR ({}), computed from prediction scores: {}'.format(config_name, average_precision_score(y_true, y_scores)))
  

    from scipy.stats import describe
    print(str(config_name) + " " + kernel + "\n--------------")
    stats = describe(results_train)
    print("Reference data\n--------------\n min={:.3f},max={:.3f},mean={:.3f},var={:.3f}".format(stats.minmax[0],
                                                                                                 stats.minmax[1],
                                                                                                 stats.mean,
                                                                                                 stats.variance))
    print("")
    stats = describe(results_novelty)
    print("Novelty data\n--------------\n min={:.3f},max={:.3f},mean={:.3f},var={:.3f}".format(stats.minmax[0],
                                                                                               stats.minmax[1],
                                                                                               stats.mean,
                                                                                               stats.variance))
    
    
    
def compactness(numpy_samples, title, print=print):
    if numpy_samples.shape[1] > 50000:
        n = 150
        print("Running compactness check with", n ,"principal components")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n)
        numpy_samples = pca.fit_transform(numpy_samples)
    clusterd = numpy_samples
    cov = np.cov(clusterd.T)
    eig = np.linalg.eigvals(cov)
    plt.imshow(cov, cmap="gray")
    plt.title(title + " (Covariance matrix)")
    plt.show()
    plt.hist(eig)
    plt.title(title + " (Eigenvalues)")
    plt.show()
    print(title + " max eigenvalue: {:.2f}, avg. l2-norm: {:.2f}".format(float(np.max(eig)), np.linalg.norm(clusterd, axis=1, ord=2).mean()))

def feature_fittness(class_a, class_b, print=print):
    if class_a.shape[1] > 50000:
        n = 150
        print("Running bayes check with", n ,"principal components")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n)
        class_a = pca.fit_transform(class_a)
        class_b = pca.fit_transform(class_b)
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    X = np.vstack((class_a, class_b))
    y = np.zeros(X.shape[0])
    y[0:class_a.shape[0]] = 1
    clf.fit(X, y)
    print("Naive-Bayes Classification:" , clf.score(X, y))
    

import numpy as np
def percentile_decision_function(train_data_pred, test_data_pred, nu=0.001, anomaly_location='left'):

    if anomaly_location == 'left':
        lower_bound = float(np.percentile(np.asarray(train_data_pred), nu * 100))
        is_anomaly = test_data_pred < lower_bound
    elif anomaly_location == 'right':
        upper_bound = float(np.percentile(np.asarray(train_data_pred), (1 - nu) * 100))
        is_anomaly = test_data_pred > upper_bound
    elif anomaly_location == 'both':
        lower_bound = float(np.percentile(np.asarray(train_data_pred), (nu / 2) * 100))
        upper_bound = float(np.percentile(np.asarray(train_data_pred), (1 - nu / 2) * 100))
        is_anomaly = np.logical_or(test_data_pred < lower_bound, test_data_pred > upper_bound)

    return is_anomaly.astype(int)


from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
def characteristic_polynomial():
    
    def cp(x):
         assert x.shape[0] == 2 and x.ndim == 1
         return np.linalg.det(x * np.eye(2) - np.array([[3,2],[1,2]]))    
    
    X = np.arange(0,100,0.1)
    Y = np.arange(0,100,0.1)
    Z = np.zeros(int((100*10)**2)).reshape(-1, int((100*10)))
    
    i = 0
    for x in X:
        j = 0
        for y in Y:
            Z[i,j] = cp(np.array([x,y]))
            j = j+1
        i = i+1
    
    im = imshow(Z,cmap=cm.RdBu) # drawing the function
    # adding the Contour lines with labels
    cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    colorbar(im) # adding the colobar on the right
    show()