
# Split data in batches
def make_batches(array_of_sites, n_cols=9):
    X_seqs = []
    y_seqs = []
    batches = []
    for site in array_of_sites:
        values = site.to_numpy()
        X_values = values[:, 0:n_cols]
        y_values = values[:, n_cols]

        X_seqs.append(X_values)
        y_seqs.append(y_values)

    return X_seqs, y_seqs