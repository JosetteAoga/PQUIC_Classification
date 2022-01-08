import csv
import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
np.set_printoptions(suppress=True)


df = pd.read_feather("../data/net_stat_data.feather")
# uncomment this line if you want to remove the instances without plugin
# df = df[df.plugin != 'no_plugin']

# create X matrix and y vector
X = df[df.columns.to_list()[:-2]] # the dataframe columns end by plugin and target
y = df.target

# add some ratio to X matrix
X["ratio_pkts"] = X.c_pkts_all / X.s_pkts_all
X["ratio_bytes"] = X.c_bytes_all / X.s_bytes_all

# the different depths to test
depths = [3,4,5]
n_instances_per_class = 3000
n_folds = 5

skf = StratifiedKFold(n_splits=n_folds)
fold_indexes = list(skf.split(X, y))

with open("../results/results_shallow.csv", "w") as stream:
  
  csv_writer = csv.writer(stream, delimiter=',')
  # write header
  csv_writer.writerow(["depth","train_fold","test_fold","train","test","confusion","mean_confusion","feat_importances", "mean_feat_importances"])
  
  for depth in depths:
    
    print("Start depth", depth)
    
    # create classifier and run cross validation
    clf = DecisionTreeClassifier(max_depth=depth)
    cv_results = cross_validate(clf, X, y, cv=iter(fold_indexes), n_jobs=-1, scoring='accuracy', return_train_score=True, return_estimator=True, verbose=10)
    print("train score:", cv_results['train_score'], np.mean(cv_results['train_score']))
    print("test score:", cv_results['test_score'], np.mean(cv_results['test_score']), "\n")

    # get confusion matrix for each fold
    confusion_str, confusions = "", []
    importance_str, importances = "", []
    for k, (train_index, test_index) in enumerate(fold_indexes):
      print("confusion on fold", k+1)
      X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      # use the learnt estimator for fold k to re-predict on test set in order to get confusion matrix
      y_pred = cv_results['estimator'][k].predict(X_test)
      confusions.append(confusion_matrix(y_test, y_pred))
      confusion_str += str(confusions[-1]) + "\n\n"

      # get the feature importances of estimator of fold k
      importances.append(cv_results['estimator'][k].feature_importances_)
      importance_str += str(importances[-1]) + "\n\n"

      # export the tree to image
      classes = ["fec", "monitoring", "multipath", "no_plugin"]
      # uncomment this line if you don't use flows without plugin
      # classes = ["fec", "monitoring", "multipath"]
      dot_data = tree.export_graphviz(cv_results['estimator'][k], out_file=None,
                                      class_names=classes,
                                      feature_names=X.columns.to_list(),
                                      filled=True, rounded=True,
                                      special_characters=False)
      graph = graphviz.Source(dot_data)
      filename = "dt_shallow_{}_net_fold_{}".format(str(depth), str(k+1))
      graph.render("../charts/shallow_net/" + filename)
    
    print("End depth", depth, "\n\n")

    test_instances_per_fold = n_instances_per_class / n_folds
    mean_confusion = (np.mean( np.array(confusions), axis=0 ) / test_instances_per_fold * 100).round(2)
    mean_confusion = str(mean_confusion)

    mean_importance = str(np.mean( np.array(importances), axis=0 ))

    csv_writer.writerow([str(depth), ",".join(str(v) for v in cv_results['train_score']), ",".join(str(v) for v in cv_results['test_score']), str(np.mean(cv_results['train_score'])), str(np.mean(cv_results['test_score'])), confusion_str, mean_confusion, importance_str, mean_importance])
    stream.flush()



