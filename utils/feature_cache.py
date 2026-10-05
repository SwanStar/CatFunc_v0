import os
import pandas as pd


class FeatureCache:
    """
    Loads raw ESM2 embedding data from CSV cache into memory and provides
    fast indexed access without repeated disk I/O.

    Usage:
        cache = FeatureCache(csv_dir, id_list, layers)
        X_all = cache.get_all(layer)          # (n_sequences, n_features)
        X_sub = cache.by_indices(idx, layer)  # X_all[idx]
    """

    def __init__(self, csv_dir, id_list, layers):
        self.csv_dir = csv_dir
        self.id_to_idx = {sid: i for i, sid in enumerate(id_list)}
        self.layers = layers
        self.all_layers = {}

        print(f"  Loading {len(layers)} layers for {len(id_list)} sequences...")
        for layer in layers:
            arr = self._load_layer_array(csv_dir, layer, id_list)
            if arr is not None:
                self.all_layers[layer] = arr
        print(f"  Loaded {len(self.all_layers)} layers "
              f"(shape per layer: {list(self.all_layers.values())[0].shape})")

    @staticmethod
    def _load_layer_array(csv_dir, layer, id_list):
        csv_path = os.path.join(csv_dir, f"layer_{layer}.csv")
        if not os.path.exists(csv_path):
            return None
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return None
        valid_cols = [sid for sid in id_list if sid in df.columns]
        arr = df[valid_cols].values.T
        return arr

    def get_all(self, layer):
        """Return (n_sequences, n_features) for the full set, or None."""
        return self.all_layers.get(layer)

    def by_indices(self, idx_array, layer):
        """Return X[idx_array, :] for a given layer."""
        X = self.all_layers.get(layer)
        return X[idx_array] if X is not None else None
