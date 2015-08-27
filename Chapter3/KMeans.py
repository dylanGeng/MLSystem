import sklearn.datasets

MLCOMP_DIR = r"C:\Users\yangwh\Documents\Python\MLSystem\Chapter3\data"

data = sklearn.datasets.load_mlcomp("20news-18828", mlcomp_root=MLCOMP_DIR)
print(data.filenames)