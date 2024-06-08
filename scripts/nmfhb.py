import os
import random
import shelve, pickle
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import describe
from sklearn.decomposition import PCA, NMF
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
import hb_dic, nmf_utils
import json

with open("notebooks/hbinfo.json", "r") as f:
    hb_config = json.load(f)

species = ["DIC", "Ca", "Mg", "K", "Na", "SO4", "Cl", "NO3", "SiO2"]
normalizer = "Na"
bootstrap = 1500
random_seed = 42

nmf_config = dict(
    init="random",
    random_state=random_seed,
    max_iter=10000,
    tol=1e-6,
)

multinmf_config = dict(
    nmf_type=nmf_utils.TrivialRescaledNMF,
    n_runs=10000,
)
n_selected = 50

heatmap_fig_config = dict(
    figsize=(8, 4),
    constrained_layout=True,
)

for watershed in hb_config["watersheds"]:
    print("Watershed:", watershed)

    df = hb_dic.load_watershed_data(
        watershed, species, filepath="data/HB_weekly_stream_chem/watersheds.xlsx"
    )
    df = df["2000":"2020"]  # Shaughnessy et al use 2000-2017


    # ------------------------------------------------------------

    preprocessor = nmf_utils.NMFPreprocessor(
        normalizer="Na", bootstrap=bootstrap, bootstrap_random_state=random_seed
    )
    V = preprocessor.transform(df)

    pca = PCA()
    pca.fit(V)
    print("Explained varience ratio:", pca.explained_variance_ratio_, sep="\n")
    n_endmember = nmf_utils.count_endmember(pca)
    print("Endmember number (explain >90% ratio):", n_endmember)

    warnings.filterwarnings("ignore")
    multi_nmf = nmf_utils.MinStdPickedNMF(
        n_selected=n_selected,
        n_components=n_endmember,
        **multinmf_config,
        **nmf_config,
    )
    Hs = multi_nmf.fit_transform(V)
    warnings.resetwarnings()


    permuter = nmf_utils.NMFKmeansPermuter(n_endmember=n_endmember, n_init=10)

    permuter.fit_transform(Hs, inplace=True)
    labels = permuter.labels_
    H_mean = Hs.mean(axis=0)
    #H_mean *= preprocessor.scaler_.values
    H_mean = pd.DataFrame(H_mean, columns=V.columns)

    # ------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = "models/output/{}/".format(timestamp)
    os.mkdir(new_dir)

    config_dict = dict(
        watershed=watershed,
        species=species,
        normalizer=normalizer,
        bootstrap=bootstrap,
        random_seed=random_seed,
        n_endmember=n_endmember,
        nmf_config=nmf_config,
        n_multinmf=multinmf_config["n_runs"],
        n_selected=n_selected,
        heatmap_fig_config=heatmap_fig_config,
        timestamp=timestamp,
    )

    with open(new_dir + "meta.txt", "w") as f:
        json.dump(config_dict, f)

    with shelve.open(new_dir + "data") as db:
        db["raw"] = Hs
        db["mean"] = H_mean

    plt.figure(**heatmap_fig_config)
    heatmap = nmf_utils.ChemistryHeatmap()
    for i, H in enumerate(Hs):
        heatmap.plot(H, V.columns)
        plt.savefig(new_dir + "heatmap_{}.png".format(i))
        plt.clf()
    heatmap.plot(H_mean, H_mean.columns)
    plt.savefig(new_dir + "heatmap_mean.svg")
