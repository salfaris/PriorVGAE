import numpy as np

class NoScaler:
    def __init__(self):
        return
    def fit(self, x):
        pass
    def fit_transform(self, x):
        return self.transform(x)
    def transform(self, x):
        return np.array(x)
    def inverse_transform(self, x):
        return np.array(x)
    
class LinearScaler:
    def __init__(self, scale: int = 1.0):
        self.scale = scale
        
    def fit(self, x):
        pass
    def fit_transform(self, x):
        return self.transform(x)
    def transform(self, x):
        return self.scale * np.array(x)
    def inverse_transform(self, x):
        return 1.0/self.scale * np.array(x)