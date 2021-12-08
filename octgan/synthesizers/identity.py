import pandas as pd
from octgan.synthesizers.base import BaseSynthesizer

class IdentitySynthesizer(BaseSynthesizer):
    def __init__(self, dataset_name, args = None):
        pass
    
    def fit(self, train_data, categorical_columns, ordinal_columns):
        self.data = pd.DataFrame(train_data)

    def sample(self, samples):
        return self.data.sample(samples, replace=True).to_numpy().copy()
