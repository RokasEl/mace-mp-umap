def get_reducers(random_state=42):
    umap_projection = umap.UMAP(
        n_components=2,
        n_neighbors=50,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    pca_projection = PCA(n_components=2)
    return umap_projection, pca_projection


def fit_dimensionality_reduction(df, tag, feature_slice, random_state=42):
    umap_reducer, pca_reducer = get_reducers(random_state)
    descriptors = np.vstack(df["descriptor"])
    print(f"Before slice {descriptors.shape}")
    descriptors = np.vstack(df["descriptor"])[:, feature_slice]
    print(f"After slice {descriptors.shape}")
    embeddings = umap_reducer.fit_transform(descriptors)
    embeddings_pca = pca_reducer.fit_transform(descriptors)
    df[f"{tag}_umap_1"] = embeddings[:, 0]
    df[f"{tag}_umap_2"] = embeddings[:, 1]
    df[f"{tag}_pca_1"] = embeddings_pca[:, 0]
    df[f"{tag}_pca_2"] = embeddings_pca[:, 1]
    return umap_reducer, pca_reducer
