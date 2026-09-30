from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessor import preprocess_record
from src.feature_extractor import (
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    extract_features_vectorized
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lgb_champion_v4_model.txt"

class EntityResolver:
    """
    Industrial Entity Resolution Engine using LightGBM GBDT ranking and 34-feature extraction.
    """
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion model file not found at {self.model_path}")
        self.model = lgb.Booster(model_file=str(self.model_path))
        self.feature_names = FEATURE_NAMES
        self.feature_descriptions = FEATURE_DESCRIPTIONS
        
    def get_feature_importances(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Returns feature importances from the 610-tree LightGBM model.
        """
        raw_imps = self.model.feature_importance(importance_type=importance_type)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": raw_imps,
            "description": [self.feature_descriptions.get(f, "") for f in self.feature_names]
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp

    def score_pairs(self, s1_records: list[dict], cand_records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """
        Scores N pairs of (s1, candidate).
        s1_records: list of dicts with 'business_name', 'business_address'
        cand_records: list of dicts with 'source', 'business_name', 'business_address'
        Returns: (probabilities: np.ndarray, feature_matrix: np.ndarray)
        """
        s1_tuples = []
        for s in s1_records:
            norm_n, norm_a, num_k = preprocess_record(s.get("business_name", ""), s.get("business_address", ""))
            s1_tuples.append((norm_n, norm_a, num_k))
            
        cand_tuples = []
        for c in cand_records:
            c_src = c.get("source", "S2")
            norm_n, norm_a, num_k = preprocess_record(c.get("business_name", ""), c.get("business_address", ""))
            cand_tuples.append((c_src, norm_n, norm_a, num_k))
            
        X = extract_features_vectorized(s1_tuples, cand_tuples)
        probs = self.model.predict(X)
        return probs, X

    def resolve_single_query(self, query: dict, candidates: list[dict], threshold: float = 0.55) -> dict:
        """
        Resolves a single S1 query against a list of candidates.
        """
        if not candidates:
            return {
                "status": "SINGLETON",
                "matched_count": 0,
                "matches": [],
                "unmatched": []
            }
            
        s1_list = [query] * len(candidates)
        probs, X = self.score_pairs(s1_list, candidates)
        
        results = []
        for idx, (cand, p) in enumerate(zip(candidates, probs)):
            feat_row = X[idx]
            # Extract top 5 driving positive signals
            feat_attributions = []
            for f_name, f_val in zip(self.feature_names, feat_row):
                if f_val > 0.3:
                    feat_attributions.append({
                        "feature": f_name,
                        "value": float(f_val),
                        "desc": self.feature_descriptions.get(f_name, "")
                    })
            feat_attributions.sort(key=lambda x: x["value"], reverse=True)
            
            cand_res = {
                "entity_id": cand.get("entity_id", f"CAND-{idx+1}"),
                "source": cand.get("source", "S2"),
                "business_name": cand.get("business_name", ""),
                "business_address": cand.get("business_address", ""),
                "probability": float(p),
                "is_match": bool(p >= threshold),
                "true_match": cand.get("is_true_match", None),
                "notes": cand.get("notes", ""),
                "top_features": feat_attributions[:6]
            }
            results.append(cand_res)
            
        # Sort by probability descending
        results.sort(key=lambda x: x["probability"], reverse=True)
        matches = [r for r in results if r["is_match"]]
        unmatched = [r for r in results if not r["is_match"]]
        
        status = "MATCH_FOUND" if len(matches) > 0 else "SINGLETON"
        return {
            "status": status,
            "matched_count": len(matches),
            "matches": matches,
            "unmatched": unmatched
        }

    def resolve_batch_with_argmax(self, df_pairs: pd.DataFrame, threshold: float = 0.55) -> pd.DataFrame:
        """
        Applies Global 1-to-1 Bipartite Argmax matching to eliminate duplicate candidate assignments.
        df_pairs required columns: 's1_id', 's1_name', 's1_address', 'cand_id', 'cand_source', 'cand_name', 'cand_address'
        """
        s1_records = df_pairs[["s1_name", "s1_address"]].rename(
            columns={"s1_name": "business_name", "s1_address": "business_address"}
        ).to_dict("records")
        
        cand_records = df_pairs[["cand_source", "cand_name", "cand_address"]].rename(
            columns={"cand_source": "source", "cand_name": "business_name", "cand_address": "business_address"}
        ).to_dict("records")
        
        probs, _ = self.score_pairs(s1_records, cand_records)
        df_pairs = df_pairs.copy()
        df_pairs["probability"] = probs
        df_pairs["passed_threshold"] = df_pairs["probability"] >= threshold
        
        # Global Argmax: each candidate ID can belong to AT MOST ONE S1 entity (highest probability)
        # 1. Filter passed threshold
        passed = df_pairs[df_pairs["passed_threshold"]]
        # 2. Group by cand_id and keep max probability row
        if not passed.empty:
            best_idx = passed.groupby("cand_id")["probability"].idxmax()
            df_pairs["is_resolved_match"] = False
            df_pairs.loc[best_idx, "is_resolved_match"] = True
        else:
            df_pairs["is_resolved_match"] = False
            
        return df_pairs
