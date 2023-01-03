# ROC plot for best factor
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

from src.constants import directory, random_state, MODEL_PATH
from src.utils import pre_processing
from sklearn.decomposition import PCA as skpca
import matplotlib.pyplot as plt


fig = plt.figure(figsize=plt.figaspect(0.5))
A, m1, name = pre_processing(directory)
clf = SVC(kernel="linear", random_state=0, tol=1e-5)
spca = skpca(n_components=m1, random_state=random_state)
ax = fig.add_subplot(223)
steps = [('scaler', StandardScaler()), ('SPCA', spca), ('SVM', clf)]
# clf.fit(p222[:, :m1], name)  # svm fit on pca data
pipe = Pipeline(steps=steps)
pipe.fit(A, name)
prediction1 = pipe.predict(A)
print(prediction1)
finaltest = pipe.decision_function(A)
print(finaltest)
fpr, tpr, thresholds = metrics.roc_curve(name, finaltest, pos_label=2.)
plt.plot(fpr, tpr)
print(fpr, tpr, thresholds)
ax.set_ylabel('True positive rate (sensitivity)')
ax.set_xlabel('False positive rate (inverse specificity)')
plt.show()

# save the model to disk
# filename1 = 'scaler_model.sav'
# pickle.dump(scaler, open(filename1, 'wb'))
# filename2 = 'pca_model.sav'
# pickle.dump(spca, open(filename2, 'wb'))
filename = MODEL_PATH + '/metrodx_model.sav'
pickle.dump(pipe, open(filename, 'wb'))
