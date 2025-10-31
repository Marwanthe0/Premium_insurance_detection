import pandas as pd
import numpy as np
import matplotlib as plt

data = {"name": ["marwan", "ahmed", "najim"], "Marks": [98, 53, 89]}
df = pd.DataFrame(data)
arr = np.array([544,3,5,12,24])
print(arr)
print(df)
df.set_index("name",inplace=True)
