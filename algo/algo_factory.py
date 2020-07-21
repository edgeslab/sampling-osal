from .network_sampling import NetworkSampling
from .non_network_sampling import NonNetworkSampling
from .hybrid_sampling import HybridSampling
from .wls_sampling import WLS_Sampling

from .cc_alfnet import CC_ALFNET

from .wvrn import wvRN
from .cc import CC
from .sgc import SGC
from .gsage import GSAGE


class AlgoFactory:

    CLF_MAP = {
        'wvrn'  :   wvRN,
        'cc'    :   CC,
        'sgc'   :   SGC,
        'gsage' :   GSAGE
    }

    SAMPLING_MAP = {
        'ns_dc_h'   :   {'method' : NetworkSampling.degree_centrality, 'params': {}},
        'ns_ct_h'   :   {'method' : NetworkSampling.clustering_coefficient, 'params': {}},
        'es_rs'     :   {'method' : NetworkSampling.edge_sampling, 'params': {}},
        'ss'        :   {'method' : NetworkSampling.snowball_sampling, 'params': {}},
        'ffs'       :   {'method' : NetworkSampling.forest_fire_sampling, 'params': {'p' : 0.7}},
        'ms'        :   {'method' : NetworkSampling.modularity_clustering, 'params': {'k' : None}},

        'rs'        :   {'method' : NonNetworkSampling.random, 'params': {}},
        'kms'       :   {'method' : NonNetworkSampling.kmeans, 'params': {'k': None, 'features': None}},

        'fp'        :   {'method' : HybridSampling.feat_prop, 'params': {'features': None}},

        'wls_2'     :   {'method' : WLS_Sampling.wls, 'params': {'k': None, 'nhops': 2}},
        'wls_3'     :   {'method' : WLS_Sampling.wls, 'params': {'k': None, 'nhops': 3}},
    }

    @staticmethod
    def get_algo(data, clf_name, sampling_name):
        if sampling_name == 'alfnet':
            return CC_ALFNET(data)
        else:
            smethod = AlgoFactory.SAMPLING_MAP[sampling_name]['method']
            sparams = AlgoFactory.SAMPLING_MAP[sampling_name]['params']
            return AlgoFactory.CLF_MAP[clf_name](data, sampling=smethod, sampling_params=sparams)