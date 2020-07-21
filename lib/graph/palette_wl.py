import math
import pdb
import numpy as np


def primes(n): 
    if n==2: return [2]
    elif n<2: return []
    s=list(range(3,n+1,2))
    mroot = n ** 0.5
    half=(n+1)/2-1
    i=0
    m=3
    while m <= mroot:
        if s[i]:
            j=int((m*m-3)/2)
            s[j]=0
            while j<half:
                s[j]=0
                j+=m
        i=i+1
        m=2*i+3
    return [2]+[x for x in s if x]


def wl_transformation(A, labels):

    # the ith entry is equal to the 2^(i-1)'th prime
    primes_arguments_required = np.asarray([2, 3, 7, 19, 53, 131, 311, 719, 1619, 3671, 8161, 17863, 38873, 84017, 180503, 386093, 821641, 1742537, 3681131, 7754077, 16290047])
    num_labels = max(labels)

    # generate enough primes to have one for each label in the graph
    try:
        prime_index = int(math.ceil(math.log(num_labels, 2)))
    except ValueError:
        pdb.set_trace()
    primes_argument = primes_arguments_required[prime_index]
    p = primes(primes_argument)

    log_primes = np.log(p[:num_labels]).T
    log_labels = np.array(list(map(lambda x: log_primes[x-1], labels)))

    Z = math.ceil(sum(log_labels))

    signatures = labels + np.matmul(A, log_labels) / Z

    # map signatures to integers counting from 1
    unique_signatures = np.unique(signatures)
    def closest(arr, element):
        return np.where(abs(arr - element) <= 1e-6)[0][0]
    new_labels = np.array(list(map(lambda x: closest(unique_signatures, x)+1, signatures)))

    return new_labels


def palette_wl(A, labels=np.array([])):
    '''
    Usage: palette_wl for labeling enclosing subgraphs in WLNM
    --Input--
    -A: original adjacency matrix of the enclosing subgraph
    -labels: initial colors
    --Output--
    -equivalence_classes: final labels

    Original author: Roman Garnett
    Original paper:
    Kersting, K., Mladenov, M., Garnett, R., and Grohe, M. Power
    Iterated Color Refinement. (2014). AAAI Conference on Artificial
    Intelligence (AAAI 2014).
    
    *author: Muhan Zhang, Washington University in St. Louis
    '''

    # if no labels given, use initially equal labels
    if not labels.any():
        labels = np.ones(A.shape[0], dtype=int)
    
    equivalence_classes = np.zeros(labels.shape[0])

    attemps = 0
    # iterate WL transformation until stability
    while (attemps < 20) and (not np.array_equal(labels, equivalence_classes)):
        equivalence_classes = labels
        labels = wl_transformation(A, labels)
        attemps += 1

    return equivalence_classes



def main():
    K = 10
    A = np.random.randint(low=0, high=2, size=(K, K))
    labels = np.random.randint(low=1, high=4, size=K)

    wl_labels = palette_wl(A)
    print(wl_labels)


if __name__ == "__main__":
    main()