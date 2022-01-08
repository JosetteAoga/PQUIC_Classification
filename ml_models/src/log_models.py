import os
import csv
import json
import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from scipy.sparse import save_npz, load_npz
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
np.set_printoptions(suppress=True)

with open("../results/results_tfidf.csv", "w") as stream:
  
  csv_writer = csv.writer(stream, delimiter=',')
  csv_writer.writerow(["header","algo","params","train_fold","test_fold","train","test","confusion","mean_confusion","feat_importances", "mean_feat_importances"])

  # run with header and without header
  for with_header in [True, False]:
    
    df, no_store = None, False
    if with_header:
      if not os.path.isfile("../data/matrix_stems_head.npz"):
        df, no_store = pd.read_feather("../data/stems.feather"), True
        print("data with header has been read")
    else:
      if not os.path.isfile("../data/matrix_stems_no_head.npz"):
        df, no_store = pd.read_feather("../data/stems_wo_head.feather"), True
        print("data without header has been read")
    
    if no_store:
      X = df.stems
      tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), use_idf=True)
      
      print("tfidf launched")
      X = tfidf.fit_transform(X)
      print("tfidf done.", X.shape)
      
      vocab = sorted(tfidf.vocabulary_, key=tfidf.vocabulary_.get)
      with open("../data/stems_vocab.txt" if with_header else "../data/stems_no_head_vocab.txt", "w") as file_voc:
        json.dump(vocab, file_voc)
        print("vocabulary stored")
      
      y = df.target
      np.save("../data/target.npy", y)
      print("target stored")
      
      print("saving X sparse features matrix after tfidf")
      save_npz('../data/' + ('matrix_stems_head' if with_header else 'matrix_stems_no_head') + ".npz", X)
      print("done")
      print("X shape: ", X.shape)

    else:
      if with_header:
        print("loading dataset with header")
      else:
        print("loading dataset without header")
      with open("../data/stems_vocab.txt" if with_header else "../data/stems_no_head_vocab.txt", "r") as file_voc:
        vocab = json.load(file_voc)
        print("vocabulary loading for no dimensionality reduction case")
      y = np.load("../data/target.npy", allow_pickle=True)
  

    head = 'matrix_stems_head' if with_header else 'matrix_stems_no_head'
    filename = head + ".npz"

    print("loading X matrix")
    X = load_npz('../data/' + filename)
    print("X shape: ", X.shape)
    if with_header:
      os.system("echo tfidf with header shape \{} >> dim.txt".format(str(X.shape).replace(')','\)')))
    else:
      os.system("echo tfidf with no header shape \{} >> dim.txt".format(str(X.shape).replace(')','\)')))

    ## DT
    mod_dt = "DT"
    clf_dt = DecisionTreeClassifier()
    pgrid_dt = {
        'max_depth': list(range(3, 8))
    }

    ## SVM
    mod_svm = "SVM"
    clf_svm = SVC(max_iter=2000, tol=0.01)
    pgrid_svm = {
        'kernel': ['poly', 'rbf'],
        'gamma': ['scale', 'auto'],
        'degree': [1, 2],
        'C': [1, 10]
    }

    mod_mnb = "MNB"
    clf_mnb = MultinomialNB()
    pgrid_mnb = {
        'alpha': [0, 0.5, 1]
    }

    n_instances_per_class = 3000
    n_folds = 5

    skf = StratifiedKFold(n_splits=n_folds)
    fold_indexes = list(skf.split(X, y))

    # loop in each classifier and its parameters for grid-search
    for name, clf, pgrid in zip([mod_dt, mod_mnb, mod_svm], [clf_dt,  clf_mnb, clf_svm], [pgrid_dt, pgrid_mnb, pgrid_svm]):

      print("header:", with_header, "algo:", name)
      gs = GridSearchCV(clf, pgrid, cv=iter(fold_indexes), n_jobs=15, scoring='accuracy', return_train_score=True, refit=False, verbose=10)
      gs.fit(X, y)

      # run the grid search CV
      # print("different params:", gs.cv_results_['params'])
      best_train_scores = list(map(lambda x : round(gs.cv_results_['split{}_train_score'.format(x)][gs.best_index_],4),list(range(n_folds))))
      best_test_scores = list(map(lambda x : round(gs.cv_results_['split{}_test_score'.format(x)][gs.best_index_],4),list(range(n_folds))))
      mean_train_score = round(np.mean(best_train_scores),4)
      mean_test_score = round(np.mean(best_test_scores),4)
      print("best params:", gs.best_params_)
      print("best score", gs.best_score_)
      print("train score:", gs.cv_results_['mean_train_score'][gs.best_index_])
      print("test score:", gs.cv_results_['mean_test_score'][gs.best_index_])

      model = None
      if name == "DT":
        model = DecisionTreeClassifier(**gs.best_params_)
      elif name == "SVM":
        model = SVC(**gs.best_params_)
      elif name == "MNB":
        model = MultinomialNB(**gs.best_params_)
      
      print("re-run model")
      cv_results = cross_validate(model, X, y, cv=iter(fold_indexes), return_estimator=True, n_jobs=n_folds)
      # confusion_str = "\n\n".join([str(confusion_matrix(y[fold_indexes[k][1]], estim.predict(X[fold_indexes[k][1]]))) for k, estim in enumerate(cv_results['estimator'])])

      # get confusion matrix for each fold
      confusion_str, confusions = "", []
      importance_str, importances = "", []
      for k, (train_index, test_index) in enumerate(fold_indexes):
        print("confusion on fold", k+1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # use the learnt estimator for fold k to re-predict on test set in order to get confusion matrix
        y_pred = cv_results['estimator'][k].predict(X_test)
        confusions.append(confusion_matrix(y_test, y_pred))
        confusion_str += str(confusions[-1]) + "\n\n"

        # get the feature importances of estimator of fold k depending on the classifier
        if name == "DT":
          importances.append(cv_results['estimator'][k].feature_importances_)
          importance_str += str(importances[-1]) + "\n\n"
      
        # export the tree to image in case of decision tree
        if name == "DT":
          dot_data = tree.export_graphviz(cv_results['estimator'][k], out_file=None, 
                                          feature_names=list(map(lambda x: "val['{}']".format(x), vocab)),
                                          class_names=["fec", "monitoring", "multipath", "no_plugin"],  
                                          filled=True, rounded=True,  
                                          special_characters=False)  
          graph = graphviz.Source(dot_data) 
          filename = "dt_with_{}header_fold_{}".format('no_' if not with_header else '', str(k+1))
          graph.render("../charts/tf_idf/" + filename)  
      print("\n\n")

      test_instances_per_fold = n_instances_per_class / n_folds
      mean_confusion = (np.mean( np.array(confusions), axis=0 ) / test_instances_per_fold * 100).round(2)
      mean_confusion = str(mean_confusion)

      if name == "DT":
        mean_importance = str(np.mean( np.array(importances), axis=0 ))
      else:
        mean_importance = ""
      
      csv_writer.writerow(["1" if with_header else "0", name, str(gs.best_params_), ",".join(str(v) for v in best_train_scores), ",".join(str(v) for v in best_test_scores), str(mean_train_score), str(mean_test_score), confusion_str, mean_confusion, importance_str, mean_importance])
      stream.flush()
