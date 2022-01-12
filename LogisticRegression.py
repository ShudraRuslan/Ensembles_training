import Imports as imp


def visualizationData_1(X, y_true):
    df = imp.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y_true))
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
    fig, ax = imp.plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    imp.plt.show()


def visualizationData_2(X, y_true):
    df = imp.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y_true))
    colors = {1: 'red', 0: 'blue'}
    fig, ax = imp.plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    imp.plt.show()


def createDecisionBoundaries1(model):
    x_net = imp.np.zeros((800, 1100))
    for i in range(800):
        for j in range(1100):
            if j % 2 == 0:
                x_net[i][j] = -4 + 0.01 * i
            else:
                x_net[i][j] = -1 + 0.01 * j
    x_net = x_net.reshape(440000, 2)
    y_net = model.predict(x_net)
    visualizationData_1(x_net, y_net)
    return model


def createDecisionBoundaries2(model):
    x_net = imp.np.zeros((600, 500))
    for i in range(600):
        for j in range(500):
            if j % 2 == 0:
                x_net[i][j] = -3 + 0.01 * i
            else:
                x_net[i][j] = -3 + 0.01 * j
    x_net = x_net.reshape(150000, 2)
    y_net = model.predict(x_net)
    visualizationData_2(x_net, y_net)
    return model


def plotROC(model, testX, testy):
    yhat = model.predict_proba(testX)
    # retrieve just the probabilities for the positive class
    pos_probs = yhat[:, 1]
    # plot no skill roc curve
    imp.plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve for model
    fpr, tpr, _ = imp.roc_curve(testy, pos_probs)
    # plot model roc curve
    imp.plt.plot(fpr, tpr, marker='.', label='model')
    # axis labels
    imp.plt.xlabel('False Positive Rate')
    imp.plt.ylabel('True Positive Rate')
    # show the legend
    imp.plt.legend()
    # show the plot
    imp.plt.show()


def plotAUCandF1FromNDepend1(model, x_test, y_test, x_train, y_train, p='f1'):
    val = imp.np.zeros(100)
    x = imp.np.zeros(100)
    if p == 'auc':
        for i in range(1, 101):
            ensModel = imp.BaggingClassifier(base_estimator=model, n_estimators=i).fit(x_train, y_train)
            y_pred = ensModel.predict(x_test)
            val[i - 1] = imp.roc_auc_score(y_test, y_pred)
            x[i - 1] = i
        imp.plt.plot(x, val, label='ensamble')
        imp.plt.xlabel('n_estimators')
        imp.plt.ylabel('AUC')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y = imp.np.zeros(100)
        for i in range(100):
            y[i] = imp.roc_auc_score(y_test, y_pred)
        imp.plt.plot(x, y, linestyle='--', label='model')
        imp.plt.show()
    else:
        for i in range(1, 101):
            ensModel = imp.BaggingClassifier(base_estimator=model, n_estimators=i).fit(x_train, y_train)
            y_pred = ensModel.predict(x_test)
            val[i - 1] = imp.f1_score(y_test, y_pred)
            x[i - 1] = i
        imp.plt.plot(x, val, label='ensamble')
        imp.plt.xlabel('n_estimators')
        imp.plt.ylabel('F1')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y = imp.np.zeros(100)
        for i in range(100):
            y[i] = imp.f1_score(y_test, y_pred)
        imp.plt.plot(x, y, linestyle='--', label='model')
        imp.plt.show()


def plotAUCandF1FromNDepend2(model, x_test, y_test, x_train, y_train, p='f1'):
    val = imp.np.zeros(20)
    x = imp.np.zeros(20)
    if p == 'auc':
        for i in range(1, 21):
            ensModel = imp.BaggingClassifier(base_estimator=model, n_estimators=i).fit(x_train, y_train)
            y_pred = ensModel.predict(x_test)
            val[i - 1] = imp.roc_auc_score(y_test, y_pred)
            x[i - 1] = i
        imp.plt.plot(x, val, label='ensamble')
        imp.plt.xlabel('n_estimators')
        imp.plt.ylabel('AUC')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y = imp.np.zeros(20)
        for i in range(20):
            y[i] = imp.roc_auc_score(y_test, y_pred)
        imp.plt.plot(x, y, linestyle='--', label='model')
        imp.plt.show()
    else:
        for i in range(1, 21):
            ensModel = imp.BaggingClassifier(base_estimator=model, n_estimators=i).fit(x_train, y_train)
            y_pred = ensModel.predict(x_test)
            val[i - 1] = imp.f1_score(y_test, y_pred)
            x[i - 1] = i
        imp.plt.plot(x, val, label='ensamble')
        imp.plt.xlabel('n_estimators')
        imp.plt.ylabel('F1')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y = imp.np.zeros(20)
        for i in range(20):
            y[i] = imp.f1_score(y_test, y_pred)
        imp.plt.plot(x, y, linestyle='--', label='model')
        imp.plt.show()


def plotAccurFromNDepend1(model, x_test, y_test, x_train, y_train, ):
    val = imp.np.zeros(100)
    x = imp.np.zeros(100)
    for i in range(1, 101):
        ensModel = imp.BaggingClassifier(base_estimator=model, n_estimators=i).fit(x_train, y_train)
        y_pred = ensModel.predict(x_test)
        val[i - 1] = imp.accuracy_score(y_test, y_pred)
        x[i - 1] = i
    imp.plt.plot(x, val, label='ensamble')
    imp.plt.xlabel('n_estimators')
    imp.plt.ylabel('Accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y = imp.np.zeros(100)
    for i in range(100):
        y[i] = imp.accuracy_score(y_test, y_pred)
    imp.plt.plot(x, y, linestyle='--', label='model')
    imp.plt.show()


def plotAccurFromNDepend2(model, x_test, y_test, x_train, y_train, ):
    val = imp.np.zeros(20)
    x = imp.np.zeros(20)
    for i in range(1, 21):
        ensModel = imp.BaggingClassifier(base_estimator=model, n_estimators=i).fit(x_train, y_train)
        y_pred = ensModel.predict(x_test)
        val[i - 1] = imp.accuracy_score(y_test, y_pred)
        x[i - 1] = i
    imp.plt.plot(x, val, label='ensamble')
    imp.plt.xlabel('n_estimators')
    imp.plt.ylabel('Accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y = imp.np.zeros(20)
    for i in range(20):
        y[i] = imp.accuracy_score(y_test, y_pred)
    imp.plt.plot(x, y, linestyle='--', label='model')
    imp.plt.show()


def calculateMetrics(y_true, y_pred, modelNumber):
    if (modelNumber == 1):
        print('Recall score is: ', imp.recall_score(y_true, y_pred, average="macro"))
        print('Precision score is: ', imp.precision_score(y_true, y_pred, average="macro"))
        print('F1 score is: ', imp.f1_score(y_true, y_pred, average="macro"))
        print('Confusion matrix: \n', imp.confusion_matrix(y_true, y_pred))
    else:
        print('Recall score is: ', imp.recall_score(y_true, y_pred))
        print('Precision score is: ', imp.precision_score(y_true, y_pred))
        print('F1 score is: ', imp.f1_score(y_true, y_pred))
        print('Confusion matrix: \n', imp.confusion_matrix(y_true, y_pred))


def testNeuralNetwork(X_train, X_test, y_train, y_test, modelNumber, learning_rate, early_stopping=False):
    model = imp.MLPClassifier(solver='sgd', learning_rate=learning_rate, early_stopping=early_stopping).fit(X_train,
                                                                                                            y_train)
    y_pred = model.predict(X_test)
    print('Score for train set: ', model.score(X_train, y_train))
    print('Score for test set: ', model.score(X_test, y_test))
    calculateMetrics(y_test, y_pred, modelNumber)
    if modelNumber == 2:
        plotROC(model, X_test, y_test)
        print('AUC is:', imp.roc_auc_score(y_test, y_pred))


def testTrees(X_train, X_test, y_train, y_test, modelNumber, max_depth, max_features, min_samples_split):
    model = imp.ExtraTreesClassifier(max_depth=max_depth, max_features=max_features,
                                     min_samples_split=min_samples_split).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Score for train set: ', model.score(X_train, y_train))
    print('Score for test set: ', model.score(X_test, y_test))
    calculateMetrics(y_test, y_pred, modelNumber)
    if modelNumber == 2:
        plotROC(model, X_test, y_test)
        print('AUC is:', imp.roc_auc_score(y_test, y_pred))


def testAnsambles(X_train, X_test, y_train, y_test, modelNumber, baseEstimator, n_estimators):
    model = imp.BaggingClassifier(base_estimator=baseEstimator, n_estimators=n_estimators).fit(X_test, y_test)
    y_pred = model.predict(X_test)
    print('Score for train set: ', model.score(X_train, y_train))
    print('Score for test set: ', model.score(X_test, y_test))
    calculateMetrics(y_test, y_pred, modelNumber)
    if modelNumber == 2:
        plotROC(model, X_test, y_test)
        print('AUC is:', imp.roc_auc_score(y_test, y_pred))
        createDecisionBoundaries2(model)
    else:
        createDecisionBoundaries1(model)


def testSimpleModel(model, X_train_1, y_train_1, X_train_2, y_train_2, X_test_1, y_test_1, X_test_2, y_test_2):
    model.fit(X_train_1, y_train_1)
    y_pred1 = model.predict(X_test_1)
    print('Score for train set: ', model.score(X_train_1, y_train_1))
    print('Score for test set: ', model.score(X_test_1, y_test_1))
    calculateMetrics(y_test_1, y_pred1, 1)
    createDecisionBoundaries1(model)
    model.fit(X_train_2, y_train_2)
    y_pred2 = model.predict(X_test_2)
    print('Score for train set: ', model.score(X_train_2, y_train_2))
    print('Score for test set: ', model.score(X_test_2, y_test_2))
    calculateMetrics(y_test_2, y_pred2, 2)
    plotROC(model, X_test_2, y_test_2)
    print('AUC is:', imp.roc_auc_score(y_test_2, y_pred2))
    createDecisionBoundaries2(model)


X_1, y_true_1 = imp.make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
rng = imp.np.random.RandomState(13)
imp.np.random.seed(0)
X_2 = imp.np.random.randn(300, 2)
y_true_2 = imp.np.logical_xor(X_2[:, 0] > 0, X_2[:, 1] > 0)

scaler = imp.StandardScaler().fit(X_1)
X_1 = scaler.transform(X_1)
scaler = imp.StandardScaler().fit(X_2)
X_2 = scaler.transform(X_2)

# visualizationData_1(X_1, y_true_1)
# visualizationData_2(X_2, y_true_2)

X_train_1, X_test_1, y_train_1, y_test_1 = imp.train_test_split(X_1, y_true_1, test_size=0.1)
X_train_2, X_test_2, y_train_2, y_test_2 = imp.train_test_split(X_2, y_true_2, test_size=0.1)

# testNeuralNetwork(X_train_1,X_test_1,y_train_1,y_test_1,1,'constant')
# testNeuralNetwork(X_train_1,X_test_1,y_train_1,y_test_1,1,'invscaling')
# testNeuralNetwork(X_train_1,X_test_1,y_train_1,y_test_1,1,'adaptive')

# testNeuralNetwork(X_train_2,X_test_2,y_train_2,y_test_2,2,'constant')
# testNeuralNetwork(X_train_2,X_test_2,y_train_2,y_test_2,2,'invscaling')
# testNeuralNetwork(X_train_2,X_test_2,y_train_2,y_test_2,2,'adaptive')

# testNeuralNetwork(X_train_1,X_test_1,y_train_1,y_test_1,1,'invscaling',True)
# testNeuralNetwork(X_train_2,X_test_2,y_train_2,y_test_2,2,'invscaling',True)

# testNeuralNetwork(X_train_1,X_test_1,y_train_1,y_test_1,1,'constant',True)
# testNeuralNetwork(X_train_2,X_test_2,y_train_2,y_test_2,2,'constant',True)

# testNeuralNetwork(X_train_1,X_test_1,y_train_1,y_test_1,1,'adaptive',True)
# testNeuralNetwork(X_train_2,X_test_2,y_train_2,y_test_2,2,'adaptive',True)

# testTrees(X_train_1,X_test_1,y_train_1,y_test_1,1,10,2,2)
# testTrees(X_train_1,X_test_1,y_train_1,y_test_1,1,2,2,10)
# testTrees(X_train_1,X_test_1,y_train_1,y_test_1,1,2,2,2)


# testTrees(X_train_2,X_test_2,y_train_2,y_test_2,2,10,2,2)
# testTrees(X_train_2,X_test_2,y_train_2,y_test_2,2,2,2,10)
# testTrees(X_train_2,X_test_2,y_train_2,y_test_2,2,2,2,2)
#
# testTrees(X_train_1, X_test_1, y_train_1, y_test_1, 1, 10, 1, 2)
# testTrees(X_train_2, X_test_2, y_train_2, y_test_2, 2, 10, 1, 2)

# model1 = imp.LogisticRegression()
# testAnsambles(X_train_1, X_test_1, y_train_1, y_test_1, 1, model1, 100)
# testAnsambles(X_train_2, X_test_2, y_train_2, y_test_2, 2, model1, 100)
#
# testSimpleModel(model1,X_train_1, y_train_1, X_train_2, y_train_2,X_test_1, y_test_1, X_test_2, y_test_2)
#
#
# model2 = imp.MLPClassifier()
# testAnsambles(X_train_1, X_test_1, y_train_1, y_test_1, 1, model2, 20)
# testAnsambles(X_train_2, X_test_2, y_train_2, y_test_2, 2, model2, 20)
# testSimpleModel(model2,X_train_1, y_train_1, X_train_2, y_train_2,X_test_1, y_test_1, X_test_2, y_test_2)

# plotAUCandF1FromNDepend1(model1, X_test_2, y_test_2, X_train_2, y_train_2)
# plotAUCandF1FromNDepend1(model1, X_test_2, y_test_2, X_train_2, y_train_2, 'auc')
# plotAUCandF1FromNDepend2(model2, X_test_2, y_test_2, X_train_2, y_train_2)
# plotAUCandF1FromNDepend2(model2, X_test_2, y_test_2, X_train_2, y_train_2, 'auc')

# plotAccurFromNDepend1(model1, X_test_2, y_test_2, X_train_2, y_train_2)
# plotAccurFromNDepend2(model2, X_test_2, y_test_2, X_train_2, y_train_2)
#
# model1 = imp.LogisticRegression(penalty='l2')
# testAnsambles(X_train_1, X_test_1, y_train_1, y_test_1, 1, model1, 100)
# model2=imp.MLPClassifier(learning_rate='adaptive',early_stopping=False)
# testAnsambles(X_train_1, X_test_1, y_train_1, y_test_1, 1, model2, 20)

# model1=imp.LogisticRegression(penalty='l2')
# testAnsambles(X_train_2, X_test_2, y_train_2, y_test_2, 2, model1, 74)
# model2=imp.MLPClassifier()
# testAnsambles(X_train_2, X_test_2, y_train_2, y_test_2, 2, model2, 5)

