import pandas as pd
df = pd.read_csv("data/chestxray14_three_labels.csv")
LABELS = ["Cardiomegaly","Edema","Effusion"]
df["num_findings"] = df[LABELS].sum(axis=1).astype(int)
df["num_findings_bin"] = df["num_findings"].map(lambda k: "1 finding" if k==1 else ("2+ findings" if k>=2 else "0"))
df.to_csv("data/chestxray14_three_labels.csv", index=False)
print(df["num_findings_bin"].value_counts())
