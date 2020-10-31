import numpy as np
import datajoint as dj
from tqdm import tqdm

from neuro_data.utils.measures import corr
from .schema_bridge import *
from .data_schemas import ScanDataset, MovieScan, InputResponse, ResponseKeys


schema = dj.schema('neuro_data_movie_stats', locals())


@schema
class ScanOracle(dj.Computed):
    definition = """
    # oracle computation for each scan
    -> ScanDataset
    ---
    n_neurons           : int       # number of neurons in scan
    pearson             : float     # mean test correlation
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> MovieScan.Unit
        ---
        pearson             : float     # mean test correlation
        """

    def make(self, key):
        trials = ScanDataset().valid_trials(key)
        conditions = stimulus.Condition.aggr(trials, count='count(*)') & 'count > 5'

        oracles, data = [], []
        for condition in tqdm(conditions.proj(), desc='Loading Conditions', total=len(conditions)):
            responses = np.stack((InputResponse.Response & key & condition).fetch('responses'), 0).swapaxes(1, 2)
            new_shape = (-1, responses.shape[-1])
            r = responses.shape[0]
            mu = responses.mean(axis=0, keepdims=True)
            oracle = (mu * r - responses) / (r - 1)
            oracles.append(oracle.reshape(new_shape))
            data.append(responses.reshape(new_shape))

        pearsons = corr(np.vstack(data), np.vstack(oracles), axis=0)
        unit_ids = (ResponseKeys.Unit & key).fetch('unit_id', order_by='row_id')

        self.insert1(dict(key, n_neurons=len(pearsons), pearson=np.mean(pearsons)))
        self.Unit.insert([dict(key, unit_id=u, pearson=p) for u, p in zip(unit_ids, pearsons)])
