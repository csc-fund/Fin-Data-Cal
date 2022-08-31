import numpy as np
import pandas as pd

s = pd.read_feather('data.f').iloc[:100000, :]
print(s.columns)
