import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_probe = pd.read_csv("../structures/df_probe_mapped.csv", header=0)
    probe_GBDT, probe_test = train_test_split(df_probe, test_size=0.5, random_state=42)

    probe_GBDT.to_csv("../structures/df_probe_GBDT.csv", index=False)
    probe_test.to_csv("../structures/df_probe_test.csv", index=False)