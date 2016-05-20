"""

    Classifiers and evaluation functions.

"""


from sklearn import svm,tree
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier, RandomForestClassifier
from scipy import sparse
import numpy as np
import itertools
from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelClf
from sklearn.metrics import mutual_info_score
from scipy.sparse.csgraph import minimum_spanning_tree

from classification.data_manipulation import *

from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests


# evaluation functions
accuracy_ev = lambda tp,tn,fp,fn: (tp+tn)/(tp+tn+fp+fn)
precision_ev = lambda tp,tn,fp,fn: tp/(tp+fp) if tp+fp != 0 else 0
recall_ev = lambda tp,tn,fp,fn: tp/(tp + fn) if tp+fn != 0 else 0

def f1_ev(tp,tn,fp,fn):
    prec = precision_ev(tp,tn,fp,fn)
    reca = recall_ev(tp,tn,fp,fn)
    if prec == reca == 0: return 0
    return 2*(prec*reca)/(prec + reca)


def cost_ev(fp_weight,fn_weight):
    return lambda x,y,w,z: w*fp_weight + z*fn_weight




def create_markov_data(data,order):
    """

    :param data: inputs
    :param order: the order of the markov chain
    :return: a new data with markov assumptions. num rows: len(data) - order. num cols: len(data[0])*(1 + order)
    """
    result = []
    for i in range(order,len(data)):
        row = []
        for step in range(0,order + 1):
            row += data[i-step]
        result.append(row)
    return result


def perform_multiple_comparison_stat(data1,data2, alpha=0.05):
    """

    :param data1:
    :param data2:
    :return: True if they are statistically different
    """
    mat1 = np.array(data1)
    mat2 = np.array(data2)
    comparisons = len(data1[0])
    pvals = [ttest_ind(mat1[:,i].tolist(),mat2[:,i])[1] for i in range(comparisons)]

    mult_comparison = multipletests(pvals, alpha=alpha)
    #print(mult_comparison)
    print(mult_comparison[0])
    """Version where just once is enough
    for val in mult_comparison[0]:
        if val == True:
            return True
    return False
    """
    # Version where the number of trues must exceed alpha (useful when you have A LOT of elements)
    true_counter = 0
    for val in mult_comparison[0]:
        if val == True:
            true_counter += 1

    return True if true_counter/len(mult_comparison[0]) >= alpha else False



class FakeClassifier:
    """
    class that returns always the same value
    """
    def __init__(self, value):
        self.value = value

    def predict(self,obs):
        return [self.value for o in obs]


def train_svms(observations,targets,evaluation_cost=(1,1),model='svm',markov_assumption='HMM',markov_order=None):
    """

    :param observations: our train dataset
    :param targets: multiple target variables.
    :param model: svm or structured_svm
    :param markov_assumption: either sliding window or HMM
    :param markov_order: if markov_assumption is HMM, we need to know the order.
    :return: the svm models in a list, one for each target variable
    """
    n_targets = len(targets[0])
    if markov_assumption == 'HMM':
        # we need to build the data on the go..
        # WE SUPPOSE THAT ALL THE ZERO PADDINGS HAVE ALREADY BEEN ADDED
        for i in range(len(targets)):
            # We use the real labels as previous information!
            if markov_order == 0: break # we don't need anything

            # we add every target outcome ahead markov_order times.
            for j in range(1,markov_order+1):
                if i + j < len(targets):
                    observations[i+j] += targets[i]

        # Then it is the same as sliding_window
    tars = np.array(targets)
    svms = []
    for i in range(n_targets):
        act_tar = tars[:,i].tolist()

        sv = svm.NuSVC()#class_weight={0:evaluation_cost[0],1:evaluation_cost[1]})
        # the svm cannot be trained if the outputs are all equal. In that case, we create a fake svm
        if len(set(act_tar)) > 1:
            # We want to have a balanced data set while training.
            bal_observations, bal_tar = sample_balanced_dataset(observations,act_tar) #from data_manipulation
            sv.fit(bal_observations,bal_tar)
        else:
            sv = FakeClassifier(act_tar[0])
        svms.append(sv)

    return svms


def predict_svms(observations,models,model='svm',markov_assumption='sliding_window',markov_order=None):
    if markov_assumption == 'sliding_window':
        n_targets = len(models)
        res = np.array([sv.predict(observations) for sv in models])
        res = res.transpose()
        return res.tolist()
    elif markov_assumption=='HMM':
        # We need to compute the result and then add it ahead. So, we predict one at a time.
        res = []
        for i in range(len(observations)):
            act_res = [sv.predict([observations[i]])[0] for sv in models]
            # I add the result ahead
            for j in range(1,markov_order+1):
                if i + j < len(observations):
                    observations[i+j] += act_res
            res.append(act_res)
        return res


def print_confusion_matrix(conf):
    print("Pred   T    F")
    print("Act ")
    for row_label, row in zip(["T","F"], conf):
        print('%s  [%s]' % (row_label, ' '.join('%04s' % i for i in row)))


def print_result_info(best_conf,best_cost):
    print("Confusion Matrix:")
    print_confusion_matrix(best_conf)
    print("Cost function: %f" % best_cost)
    print("Evaluations: acc: %f, prec: %f, reca: %f, f1:%f" % (accuracy_ev(best_conf[0][0],best_conf[1][1],best_conf[1][0],best_conf[0][1])
                                                               ,precision_ev(best_conf[0][0],best_conf[1][1],best_conf[1][0],best_conf[0][1]),
                                                               recall_ev(best_conf[0][0],best_conf[1][1],best_conf[1][0],best_conf[0][1]),
                                                               f1_ev(best_conf[0][0],best_conf[1][1],best_conf[1][0],best_conf[0][1])))



def test_svms(obs,truths,models):
    """

    tests the models with obs and truths and returns a confusion matrix
    """

    predictions = predict_svms(obs,models)
    tp,tn,fp,fn = 0,0,0,0
    for i, ro in enumerate(predictions):
        for j, p in enumerate(ro):
            if p == 1:
                if truths[i][j] == 1: tp += 1
                else: fp += 1
            else:
                if truths[i][j] == 0: tn += 1
                else: fn += 1
    conf = [[tp,fn],[fp,tn]]
    return conf


def compute_confusion_matrix(prediction,truths):
    tp,tn,fp,fn = 0,0,0,0
    for i, ro in enumerate(prediction):
        for j, p in enumerate(ro):
            if p == 1:
                if truths[i][j] == 1: tp += 1
                else: fp += 1
            else:
                if truths[i][j] == 0: tn += 1
                else: fn += 1
    conf = [[tp,fn],[fp,tn]]
    return conf


def evaluate_svms(train_x,train_y,test_x,test_y):
    """
    trains and tests the model with svms. returns a confusion matrix
    """
    return test_svms(test_x,test_y,train_svms(train_x,train_y))


def chow_liu_tree(y_):
    # compute mutual information using sklearn
    n_labels = y_.shape[1]
    mi = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            mi[i, j] = mutual_info_score(y_[:, i], y_[:, j])
    mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
    edges = np.vstack(mst.nonzero()).T
    edges.sort(axis=1)
    return edges


def transform_to_float64(data):
    return [[float(x) for x in line] for line in data]


def train_structured_svm(observations,targets):
    """

    :param observations: our train dataset
    :param targets: multiple target variables.
    :return: the structured svm model
    """

    # ideally you can say the edges that are connected. For now, we use full.
    n_labels = len(targets[0])

    full = np.vstack([x for x in itertools.combinations(range(n_labels), 2)])
    #tree = chow_liu_tree(targets)

    # Choose the best model...
    full_model = MultiLabelClf(edges=full, inference_method='lp')

    #tree_model = MultiLabelClf(edges=tree, inference_method="max-product")
    full_ssvm = OneSlackSSVM(full_model, inference_cache=50, C=.1, tol=0.01)
    full_ssvm.fit(np.array(observations), np.array(targets))

    return full_ssvm


def predict_structured_svm(observations,model):
    res = model.predict(np.array(observations,dtype='float64'))
    return res

def train_dts(observations,targets,method='bagging'):
    """Trains a decision tree for each output

    :param observations: our train dataset
    :param targets: multiple target variables.
    :param method: bagging,random_forest,boosting
    :return: the dt models in a list, one for each target variable
    """
    n_targets = len(targets[0])

    tars = np.array(targets)
    dts = []
    for i in range(n_targets):
        act_tar = tars[:,i].tolist()

        dt = None
        if method == 'bagging': dt = BaggingClassifier(tree.DecisionTreeClassifier(),n_estimators=100,max_samples=0.5, max_features=1.)
        elif method == 'random_forest': dt = RandomForestClassifier(n_estimators=100)
        elif method == 'boosting': dt = AdaBoostClassifier(n_estimators=100)
        else: dt = tree.DecisionTreeClassifier()
        # the dt cannot be trained if the outputs are all equal. In that case, we create a fake dt
        if len(set(act_tar)) > 1:
            # We want to have a balanced data set while training.
            bal_observations, bal_tar = sample_balanced_dataset(observations,act_tar) #from data_manipulation
            dt.fit(bal_observations,bal_tar)
        else:
            dt = FakeClassifier(act_tar[0])
        dts.append(dt)

    return dts

def predict_dts(observations,models):

    n_targets = len(models)
    res = np.array([dt.predict(observations) for dt in models])
    res = res.transpose()
    return res.tolist()



def predict_threshold(data,n_nodes,distance,INF_TOKEN):
    """

        :param data: a matrix of raw distances: (n_nodes-1) distances per node.
        :param n_nodes: the number of nodes we have in the network
        :param distance: the threshold we consider interaction
        :param INF_TOKEN: the value for 'not sensed'
        :return a matrix of interactions.
    """

    result = []
    #for every line in the data we infer the interactions:
    for line in data:
        #print(line)
        interactions = set()

        for i in range(n_nodes):
            for j in range(n_nodes-1):
                elem = line[(n_nodes-1)*i + j]
                if elem != INF_TOKEN and elem <= distance:
                    # it means we consider an interaction.
                    # i is the right first index, j however, isn't
                    if i <= j:
                        # example (0,0) are indexes (0,1), (1,2) are indexes (1,3)
                        interactions.add(tuple(sorted([i,j+1])))
                    else:
                        # example (1,0) is indeed (1,0)
                        interactions.add(tuple(sorted([i,j])))
        #print(list(interactions))
        result.append(list(interactions))

    return transform_list_to_matrix_representation(result,n_nodes)


def predict_threshold_with_time(data,n_nodes,distance,prev_times,INF_TOKEN):
    """

        :param data: a matrix of raw distances: (n_nodes-1) distances per node.
        :param n_nodes: the number of nodes we have in the network
        :param distance: the threshold we consider interaction
        :param prev_times: for how long that edge must be under threshold to be considered
        :param INF_TOKEN: the value for 'not sensed'
        :return a matrix of interactions.
    """
    if prev_times == 0: return predict_threshold(data,n_nodes,distance, INF_TOKEN)

    result = []
    #for every line in the data we infer the interactions:
    for line in data:
        #print(line)
        interactions = set()

        for i in range(n_nodes):
            for j in range(n_nodes-1):
                elem = line[(n_nodes-1)*i + j]
                if elem != INF_TOKEN and elem <= distance:
                    # it means we consider an interaction.
                    # i is the right first index, j however, isn't
                    if i <= j:
                        # example (0,0) are indexes (0,1), (1,2) are indexes (1,3)
                        interactions.add(tuple(sorted([i,j+1])))
                    else:
                        # example (1,0) is indeed (1,0)
                        interactions.add(tuple(sorted([i,j])))
        #print(list(interactions))
        result.append(list(interactions))

    #Now we have to create a new list whereto add only the edges that appear previously as well.
    adjusted_result = []
    # We can't possibly have edges at the beginning
    for _ in range(min(prev_times,len(result))):
        adjusted_result.append([])

    for i, line in enumerate(result[prev_times:]):
        new_line = []
        for edge in line:
            valid = True
            for j in range(1,prev_times+1):
                if not (edge in result[i-j]):
                    valid = False
                    break
            if valid: new_line.append(edge)
        adjusted_result.append(new_line)
    return transform_list_to_matrix_representation(adjusted_result,n_nodes)

"""
# Example usage of SVM
X = [[0, 0, 2], [1, 1, 5]]
y = [[0, 1, 0],[1, 0, 0]]

Tes = [[0, 0, 2], [1, 1, 5],[0, 0, 2], [1, 1, 5],[0, 0, 2], [1, 1, 5]]
trs = [[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]

svms = train_svms(X,y)
print(test_svms(Tes,trs,svms))
"""



