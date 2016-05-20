import random


def get_edge_index(edge,n_elems):
    """

    :param edge:
    :param n_elems:
    :return: the position of the array sorted like: [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    """
    ind = 0
    for i in range(edge[0]):
        ind += n_elems - i - 1
    ind += edge[1] - edge[0] - 1
    if ind >= n_elems*(n_elems-1)//2:
        print("Index out of bounds! edge: (%d,%d), index: %d" % (edge[0],edge[1],ind))
    return ind


def transform_list_to_matrix_representation(data,n_elems):
    """

    :param data: a list of list of edges
    :param n_elems: the number of nodes. Important because we have to make n*(n-1)/2 values
    :return: for every starting list, a list of n*(n-1)/2 values
    """
    default_row = [0]* (n_elems*(n_elems-1)//2)
    result = []
    for row in data:
        new_row = default_row[:]
        for edge in row:
            if edge[0] >= n_elems or edge[1] >= n_elems:
                print("Unexpected edge: (%d,%d)" % (edge[0],edge[1]))
                continue
            new_row[get_edge_index(edge,n_elems)] = 1
        result.append(new_row)

    return result

def sample_balanced_dataset(X,y):
    """

    :param X: input data
    :param y: list of outputs. WE SUPPOSE IT IS ONLY 0 AND 1!
    :return: a random balanced sample of the dataset. It returns ALL the elements of the minority class.
    """
    # we create a mask to split zeros and ones
    mask = [True if x == 1 else False for x in y]
    # we split our dataset:
    X1 = list(map(lambda z: z[1],filter(lambda x: mask[x[0]],enumerate(X))))
    X0 = list(map(lambda z: z[1],filter(lambda x: not mask[x[0]],enumerate(X))))

    bigger = X1 if len(X1) > len(X0) else X0
    sampling_size = min(len(X1),len(X0))

    # These elements remain unchanged
    new_X = X0[:] if len(X1) > len(X0) else X1[:]
    new_X += random.sample(bigger, sampling_size)
    # we sort these results
    new_y = [0]*sampling_size + [1]*sampling_size if len(X1) > len(X0) else [1]*sampling_size + [0]*sampling_size

    return new_X,new_y

