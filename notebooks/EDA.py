import pandas as pd


raw_path = R"C:\Users\aleja\Documents\github_repositories\cancer-clinical-test\data\Yuan_expr_clinpat.csv"
df = pd.read_csv(raw_path, sep=";")


print(df.shape)

