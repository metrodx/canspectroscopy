# Edward's code

import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.cm as cm
from os import listdir
from scipy.interpolate import interp1d
# import spc_spectra
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from sklearn.decomposition import PCA as skpca
from sklearn.preprocessing import StandardScaler

import pickle

# load the model from disk
filename1 = 'scaler_model.sav'
filename2 = 'pca_model.sav'
filename3 = 'svm_model.sav'
scaler = pickle.load(open(filename1, 'rb'))
spca = pickle.load(open(filename2, 'rb'))
clf = pickle.load(open(filename3, 'rb'))
m1 = 44

# range
lb = 800  # 3#132
ub = 1800  # 2250 #3000 #1200 #4000
# ALSS peramiters
alssp = 0.001
lnda = 50000000

fig = plt.figure(figsize=plt.figaspect(0.5))


# ALSS baseline
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


# test spectra
"D:\\Panc\\sorted\\severe pan c\\test"
"I:\\FTIR\\spray\\Panc\\t4"
"D:\\FTIR\\spray\\Panc\\cv 22"
"D:\\FTIR\\2022\\windowing\\Who test"
"D:\\FTIR\\spray\\Panc\\who s all"
directory2 = "D:\\FTIR\\2022\\windowing\\sub10"
files2 = listdir(directory2)

n2 = len(files2)  # number of input files

r22 = np.arange(0, n2, 1)
samplename2 = []
for i in (r22):
    term2 = directory2 + '\\%s' % files2[i]  #
    samplename2.append(term2[-20:])
    if term2.endswith('.csv'):
        with open(term2, 'r') as f2:
            reader2 = csv.reader(f2, delimiter=',')
            # get header from first/second rows
            headers21 = next(reader2)
            headers22 = next(reader2)
            # get all the rows as a list
            data2 = list(reader2)
            # transform data into numpy array
            data2 = np.array(data2)[:, 0:2].astype(float)
    else:
        print('file type unsupported')
    if i < 1:
        x2 = np.arange(lb, ub + 0.5, 0.5)  # x axis ideal
        r23 = np.arange(0, len(x2), 1)
        # x,z=data[w1:w2,_] if want range
        dataset2 = np.zeros((((int(n2))), len(x2)))
    xi2 = data2[:, 0]  # x axis from csv
    yi2 = data2[:, 1]  # z axis from csv

    inter = interp1d(xi2, yi2)

    regex2 = re.compile(r'\d+')
    dataset2[i, :] = 100 - inter(x2)

print(samplename2)

D22 = np.zeros((((int(n2))), len(x2)))
for i in r22:
    d22 = dataset2[i] - baseline_als(dataset2[i], lnda, alssp)
    d22 = d22 / np.average(d22)
    D22[i, :] = d22 * 1

# test the test spectra
nfeat2 = scaler.transform(D22)
D22p = spca.transform(nfeat2)
print(D22)
print(D22p)
prediction = clf.predict(D22p[:, :m1])
print(prediction)
p1a = np.average(prediction[0:15]) / 2, 1 - np.average(prediction[15:]) / 2
print(p1a)
finaltest2 = clf.decision_function(D22p[:, :m1])
print(finaltest2)

# view tests
ax = fig.add_subplot(224)
for i in r22:
    plt.plot(x2, D22[i], linewidth=0.5, label=samplename2[i])
ax.legend()

plt.show()
