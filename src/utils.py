# Edward's code

import re
import matplotlib as mpl
from matplotlib.lines import Line2D
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

from src.constants import lb, ub, lnda, alssp, number, size, L, pcn, c2, random_state


def baseline_als(y, lam, p, niter=10):
    """
    Helper function to estimate the baselines by penalizing the differences in the baseline corrected
    signals, which makes it possible to eliminate scatter effects on the spectra.

    :param y:
    :param lam:
    :param p:
    :param niter:

    :return:

    """
    l = len(y)
    d = sparse.csc_matrix(np.diff(np.eye(l), 2))
    w = np.ones(l)
    for i in range(niter):
        W = sparse.spdiags(w, 0, l, l)
        Z = W + lam * d.dot(d.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def pre_processing(directory):
    """

    :param directory:
    :return:
    """
    files = listdir(directory)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    n = len(files)  # number of input files
    print(n)
    # loo = LeaveOneOut()
    # lpo = LeavePOut(p=3)
    # clf = LinearDiscriminantAnalysis(n_components=pcn - 1)  # Linear Discriminant Analysis classification
    lpo = KFold(n_splits=int((n / 3)))
    clf = SVC(kernel="linear", random_state=0, tol=1e-5)

    # colors = cm.rainbow(np.linspace(0, 1, int(n)))  # trying to make the colours correspond for each spectra displayed
    # count4 = 0

    # n3= 5 # number of components looked at
    r = np.arange(0, n, 1)
    r2 = np.arange(0, int(n), 1)  # same as r, but I sometimes chaged this to n/2
    # support vector classification
    parameters = []  # 2242, laser wl(nm), grating, holes, slit, power, objective, time, acc, R

    sample_name = []
    name = []

    # opens 3 types of files and put's them into 'dataset' array
    for i in r:

        term = directory + '/%s' % files[i]  #
        # print (term)
        sample_name.append(term)
        if term[1 + len(directory)] == 'H':
            name.append(0)
        elif term[1 + len(directory)] == 'P':
            name.append(0)
        else:
            name.append(2)
        # name.append(term[8+len(directory)]) # for raman

        if term.endswith('.txt'):
            f = open(term, "r")
            data = []
            headers = []

            for line in f:
                data.append(line)

            f.close()
            data = [x.strip('\n') for x in data]
            data = [x.split('\t') for x in data]
            data = np.array(data).astype(float)
        elif term.endswith('.csv'):
            with open(term, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                # get header from first/second rows
                headers1 = next(reader)
                headers2 = next(reader)
                # get all the rows as a list
                data = list(reader)
                # transform data into numpy array
                data = np.array(data)[:, 0:2].astype(float)
        # elif term.endswith('.spc'):
        #     data = []
        #     f = spc_spectra.File(term)
        #     data.append(f.data_txt())
        #     data = [x.strip('\n') for x in data]
        #     data = [x.replace('\t', '\n') for x in data]
        #     data = [x.split('\n') for x in data]
        #     data = np.array(data).astype(float).reshape(-1, 2)
        else:
            print('file type unsupported')
        # no interpolation
        """  
        if i < 1 : 
            x= data[:,0]  
    
            dataset = np.zeros((((int(n))),len(x)))    
        dataset[i-1,:] = data[:,1]"""

        # section modified for interpolation (to a standard 1/2 wave number set)

        if i < 1:
            x = np.arange(lb, ub + 0.5, 0.5)  # x axis ideal
            r3 = np.arange(0, len(x), 1)
            # x,z=data[w1:w2,_] if want range
            dataset = np.zeros(((int(n)), len(x)))
        xi = data[:, 0]  # x axis from csv
        yi = data[:, 1]  # z axis from csv

        inter = interp1d(xi, yi)

        regex = re.compile(r'\d+')
        my_ints = [int(l) for l in regex.findall(term)]
        if my_ints[0] == 2242:
            my_ints = my_ints[1:]
        parameters.append(my_ints)
        if term.endswith('.csv'):
            dataset[i, :] = 100 - inter(x)
        else:
            dataset[i, :] = 100 - inter(x)

    name = np.array(name).astype(float)
    print(len(name))

    # average spectra to use as target for EMSC
    d3 = np.zeros(len(x))

    for i in r3:
        d3[i] = np.average(dataset[:, i])

    # defining reference from ALSS baselined average spectra
    d3 = d3 - baseline_als(d3, lnda, alssp)

    D2 = np.zeros(((int(n)), len(x)))

    # norm and alss
    for i in r2:
        d2 = dataset[i] - baseline_als(dataset[i], lnda, alssp)
        d2 = d2 / np.average(d2)
        D2[i, :] = d2 * 1

    ax = fig.add_subplot(222)

    # plot
    for i in r2:
        plt.plot(x, D2[i], color=c2[int(name[i])], linewidth=0.5)
        # np.savetxt("C:\\Users\\Edwar\\Downloads\\USb\\Output\\corrected%s.csv" %i, np.concatenate((x.reshape(-1,1),
        # d2.reshape(-1,1)),axis=1), delimiter=",") #save the data as csv

    plt.xlabel('wavenumber (cm-1)')
    plt.ylabel('normalised intensity')
    tit0 = 'Baselined and normalised spectra for sample:', directory[-8:]
    plt.title(tit0)

    # cross val
    results = []
    r3 = np.arange(0, number, 1)
    for i in r3:
        result = int(
            np.average(
                cross_val_score(estimator=clf, X=D2[:, i * size:(i + 1) * size], y=name, cv=lpo)[0:L]) * 100), int(
            np.average(cross_val_score(estimator=clf, X=D2[:, i * size:(i + 1) * size], y=name, cv=lpo)[L:]) * 100)
        results.append(result)
    print(results)
    print(lb + results.index(max(results)) * size)

    yl = np.arange(0, 1, 0.01)
    xl = np.zeros(len(yl)) + lb
    for i in r3:
        ax.plot(xl + i * size / 2, yl, label=results[i])
    ax.legend()

    A = D2 * 1  # A = (D2[:,results.index(max(results))*size:(results.index(max(results))+1)*size])
    dot = np.dot

    ax = fig.add_subplot(224)

    result = cross_val_score(estimator=clf, X=A, y=name, cv=lpo)
    tit = 'SVM Cross validation Sens./Spec', int(np.average(result[0:L]) * 100), int(np.average(result[L:]) * 100), '%'
    plt.title(tit)
    plt.xlabel('wave number (cm-1)')
    plt.ylabel('normalised intensity')

    # custom legend

    custom_lines = [Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4)]
    ax.legend(custom_lines, ['Healthy', 'Pre-malignant', 'Cancer'])

    # section creating average graphs and plotting them with error
    b1 = 0
    b2 = 0
    b3 = 0
    b1e, b3e, b2e = [], [], []
    b1er, b3er, b2er = [], [], []
    # add all spectra together and group for error calculation
    for i in r:
        if name[i] == 0:
            b1 = b1 + D2[i]
            b1e.append(D2[i])
        elif name[i] == 1:
            b2 = b2 + D2[i]
            b2e.append(D2[i])
        else:
            b3 = b3 + D2[i]
            b3e.append(D2[i])
    # compile errors
    r4 = np.arange(0, len(x), 1)
    print(np.sqrt(len(np.array(b3e)[:, 1])))
    for i in r4:
        if b1e: b1er.append(np.std(np.array(b1e)[:, i]) / np.sqrt(len(np.array(b1e)[:, i])))
        if b2e: b2er.append(np.std(np.array(b2e)[:, i]) / np.sqrt(len(np.array(b2e)[:, i])))
        if b3e: b3er.append(np.std(np.array(b3e)[:, i]) / np.sqrt(len(np.array(b3e)[:, i])))

    # divide sum by number and plot with errors
    if b1e:
        b1 = b1 / (list(name).count(0))
        plt.plot(x, b1, color=c2[0], linewidth=0.5)
        plt.fill_between(x, b1 - b1er, b1 + b1er,
                         color=c2[0], alpha=0.2)
    if b2e:
        b2 = b2 / (list(name).count(1))
        plt.plot(x, b2, color=c2[1], linewidth=0.5)
        plt.fill_between(x, b2 - b2er, b2 + b2er,
                         color=c2[1], alpha=0.2)
    if b3e:
        b3 = b3 / (list(name).count(2))
        plt.plot(x, b3, color=c2[2], linewidth=0.5)
        plt.fill_between(x, b3 - b3er, b3 + b3er,
                         color=c2[2], alpha=0.2)

    # np.set_printoptions(threshold=sys.maxsize)
    # b4= np.array([x,b1,b2,b3,b1er,b2er,b3er])
    # np.savetxt("I:\\Python\\Edward\\hmw ave.csv" , x, delimiter=",")
    # np.savetxt("I:\\Python\\Edward\\hmw ave1.csv" , b1, delimiter=",")
    # np.savetxt("I:\\Python\\Edward\\hmw ave2.csv" , b2, delimiter=",")
    # np.savetxt("I:\\Python\\Edward\\hmw ave3.csv" , b3, delimiter=",")
    # np.savetxt("I:\\Python\\Edward\\hmw ave4.csv" , b1er, delimiter=",")
    # np.savetxt("I:\\Python\\Edward\\hmw ave5.csv" , b2er, delimiter=",")
    # np.savetxt("I:\\Python\\Edward\\hmw ave6.csv" , b3er, delimiter=",") #save the data as csv
    # import bz.numpyutil as nu
    # dot = nu.pdot
    #

    ax = fig.add_subplot(221)

    # early jupyter bit
    spca = skpca(n_components=pcn, random_state=random_state)
    scaler = StandardScaler()
    scaler.fit(A)
    nfeat1 = scaler.transform(A)
    spca.fit(nfeat1)  # pca fit
    p222 = spca.transform(nfeat1)

    # varied PC classification accuracies plot
    rez = np.arange(0, pcn, 1)
    for i in np.arange(0, pcn, 1):
        print(i + 1)
        result3 = cross_val_score(estimator=clf, X=p222[:, :i + 1], y=name, cv=lpo)
        if i == 5:
            print(result3)
        sens = int(np.average(result3[0:L]) * 100)
        spec = int(np.average(result3[L:]) * 100)
        rez[i] = (sens + spec) / 2
        plt.scatter(i + 1, sens, color='cyan')
        plt.scatter(i + 1, spec, color='magenta')
        ax.annotate(sens, (i + 1, sens))
        ax.annotate(spec, (i + 1, spec))

    # custom legend

    custom_lines = [Line2D([0], [0], color='cyan', lw=4),
                    Line2D([0], [0], color='magenta', lw=4)]

    ax.legend(custom_lines, ['Sensitivity', 'Specificity'])

    plt.xlabel('Number of PCs')
    plt.ylabel('Accuracy')
    m1 = (np.argmax(rez) + 1)
    m2 = max(rez)
    plt.title(str(('PCA SVM accuracies. PC max:', m1, ' Accuracy:', m2, '%')).strip(','))

    return A, m1, name


def test_pre_processing(directory):
    """

    :param directory:
    :return:
    """
    files = listdir(directory)
    n = len(files)
    r22 = np.arange(0, n, 1)
    samples = []
    for i in r22:
        term2 = directory + '/%s' % files[i]  #
        samples.append(term2[-20:])
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
            dataset2 = np.zeros(((int(n)), len(x2)))
        xi2 = data2[:, 0]  # x axis from csv
        yi2 = data2[:, 1]  # z axis from csv

        inter = interp1d(xi2, yi2)

        dataset2[i, :] = 100 - inter(x2)

    D22 = np.zeros((((int(n))), len(x2)))
    for i in r22:
        d22 = dataset2[i] - baseline_als(dataset2[i], lnda, alssp)
        d22 = d22 / np.average(d22)
        D22[i, :] = d22 * 1
    return D22
