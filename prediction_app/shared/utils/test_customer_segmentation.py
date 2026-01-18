import unittest

import numpy as np
import pandas as pd

from prediction_app.shared.utils.customer_segmentation import segment_customers_kmeans


class TestCustomerSegmentation(unittest.TestCase):
    def test_basic_clustering_success(self) -> None:
        rng = np.random.default_rng(42)
        n = 60
        # Three clearly separated clusters in (Recency, Frequency, Monetary).
        c1 = rng.normal(loc=[10, 20, 500], scale=[2, 2, 30], size=(n, 3))
        c2 = rng.normal(loc=[120, 3, 80], scale=[5, 1, 10], size=(n, 3))
        c3 = rng.normal(loc=[40, 8, 200], scale=[3, 2, 20], size=(n, 3))
        X = np.vstack([c1, c2, c3])
        df = pd.DataFrame(X, columns=["Recency", "Frequency", "Monetary"])
        df.insert(0, "user_id", np.arange(df.shape[0]) + 1)

        res = segment_customers_kmeans(
            df,
            selected_features=["Recency", "Frequency", "Monetary"],
            k_range=(2, 5),
            random_seed=7,
            outlier_threshold=None,
        )
        self.assertEqual(res["status"], "success")
        self.assertIn(res["model_info"]["best_k"], [2, 3, 4, 5])
        self.assertGreater(res["model_info"]["best_silhouette_score"], 0.5)
        self.assertEqual(set(res["model_info"]["features_used"]), {"Recency", "Frequency", "Monetary"})
        self.assertTrue(len(res["clusters_summary"]) >= 2)
        self.assertTrue(len(res["elbow_curve_data"]) == 4)

    def test_zero_variance_feature_is_dropped(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5, 6],
                "a": [0, 0, 0, 10, 10, 10],
                "b": [1, 1, 1, 1, 1, 1],  # constant
            }
        )

        res = segment_customers_kmeans(df, selected_features=["a", "b"], k_range=(2, 3), random_seed=1)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_info"]["features_used"], ["a"])
        self.assertTrue(any("zero-variance" in w for w in res.get("warnings", [])))

    def test_too_few_samples_errors(self) -> None:
        df = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "Recency": [1, 2, 3, 4, 5]})
        res = segment_customers_kmeans(df, selected_features=["Recency"], k_range=(3, 6))
        self.assertEqual(res["status"], "error")
        self.assertIn("数据量不足", res["message"])


if __name__ == "__main__":
    unittest.main()

