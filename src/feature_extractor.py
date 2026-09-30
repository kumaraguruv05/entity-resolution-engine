import numpy as np
from rapidfuzz import fuzz
from anyascii import anyascii as to_ascii

from src.preprocessor import clean_domain, get_char_3grams

FEATURE_NAMES = [
    "raw_name_ratio",
    "raw_name_partial_ratio",
    "raw_name_token_sort",
    "raw_name_token_set",
    "name_jaccard",
    "name_char_3gram_jaccard",
    "name_length_ratio",
    "name_exact_match",
    "name_first4_match",
    "name_first_token_match",
    "is_indic_transliterated",
    "translit_ratio",
    "translit_token_sort",
    "translit_jaccard",
    "translit_char_3gram",
    "domain_exact_match",
    "domain_contains",
    "domain_ratio",
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard",
    "addr_char_3gram",
    "is_cand_addr_nan",
    "exact_numeric_key_match",
    "numeric_jaccard",
    "numeric_count_diff",
    "numeric_conflict",
    "first_number_match",
    "max_name_similarity",
    "max_name_jaccard",
    "high_address_anchor",
    "name_addr_harmonic_mean",
    "source_is_s2"
]

FEATURE_DESCRIPTIONS = {
    "raw_name_ratio": "Levenshtein similarity ratio between query and candidate names (0-1).",
    "raw_name_partial_ratio": "Best matching substring similarity between names.",
    "raw_name_token_sort": "Token-sorted fuzzy ratio (resilient to word order permutations).",
    "raw_name_token_set": "Token-set ratio (handles extra corporate suffixes like LLC, Inc).",
    "name_jaccard": "Word token intersection over union between names.",
    "name_char_3gram_jaccard": "Character 3-gram overlap (robust against typos and phonetic misspellings).",
    "name_length_ratio": "Ratio between shorter and longer name string lengths.",
    "name_exact_match": "Binary flag indicating 100% identical normalized name strings.",
    "name_first4_match": "Prefix match on the first 4 characters of the business name.",
    "name_first_token_match": "Exact match on the primary leading brand token.",
    "is_indic_transliterated": "Flag: 1 if candidate contains non-ASCII Indic Unicode characters.",
    "translit_ratio": "Fuzzy ratio after phonetically Romanizing Indic text via AnyAscii.",
    "translit_token_sort": "Token-sort ratio on Romanized transliterated text.",
    "translit_jaccard": "Word Jaccard similarity on Romanized transliteration tokens.",
    "translit_char_3gram": "Character 3-gram overlap on Romanized transliterated text.",
    "domain_exact_match": "Exact match after stripping web TLDs (.com, .org, .net, etc.).",
    "domain_contains": "Substring containment between cleaned domain and company name.",
    "domain_ratio": "Fuzzy similarity between stripped domain name and reference company name.",
    "addr_ratio": "Fuzzy string similarity between normalized street addresses.",
    "addr_token_sort": "Token-sort address similarity (resilient to street abbreviation ordering).",
    "addr_token_set": "Token-set address similarity (handles missing suite or floor numbers).",
    "addr_jaccard": "Word token Jaccard overlap of address tokens.",
    "addr_char_3gram": "Character 3-gram address overlap.",
    "is_cand_addr_nan": "Flag: 1 if candidate address was missing/NaN in the source database.",
    "exact_numeric_key_match": "Exact match of all street numbers and postal codes.",
    "numeric_jaccard": "Jaccard similarity of extracted numerical tokens.",
    "numeric_count_diff": "Absolute difference in count of numbers between addresses.",
    "numeric_conflict": "Conflict penalty: 1 if both records have numbers but zero overlap.",
    "first_number_match": "Exact match on the primary building/house number.",
    "max_name_similarity": "Upper envelope of raw name, transliterated name, and domain name similarities.",
    "max_name_jaccard": "Upper envelope of raw and transliterated word Jaccards.",
    "high_address_anchor": "Strong physical anchor signal: High address overlap with 0 numeric conflict.",
    "name_addr_harmonic_mean": "Harmonic mean of name and address similarity (balances both signals).",
    "source_is_s2": "Binary indicator for Source 2 (1.0) vs Source 3 (0.0)."
}

def extract_features_vectorized(s1_tuples: list, cand_tuples: list) -> np.ndarray:
    """
    Extracts the 34-feature matrix for N pairs.
    s1_tuples: list of (norm_name, norm_addr, num_key)
    cand_tuples: list of (source_str, norm_name, norm_addr, num_key)
    """
    n = len(s1_tuples)
    X = np.zeros((n, 34), dtype=np.float32)
    
    for i in range(n):
        s1_n, s1_a, s1_k = s1_tuples[i]
        c_src, c_n, c_a, c_k = cand_tuples[i]
        
        # 1. Raw Name features
        r_ratio = fuzz.ratio(s1_n, c_n) / 100.0
        r_part = fuzz.partial_ratio(s1_n, c_n) / 100.0
        r_sort = fuzz.token_sort_ratio(s1_n, c_n) / 100.0
        r_set = fuzz.token_set_ratio(s1_n, c_n) / 100.0
        
        toks1 = set(s1_n.split())
        toks2 = set(c_n.split())
        un = toks1 | toks2
        n_jac = (len(toks1 & toks2) / len(un)) if un else 0.0
        
        g1 = get_char_3grams(s1_n)
        g2 = get_char_3grams(c_n)
        gun = g1 | g2
        n_g3 = (len(g1 & g2) / len(gun)) if gun else 0.0
        
        l1, l2 = len(s1_n), len(c_n)
        len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0
        exact_n = 1.0 if (s1_n and s1_n == c_n) else 0.0
        first4_n = 1.0 if (l1 >= 4 and l2 >= 4 and s1_n[:4] == c_n[:4]) else 0.0
        first_tok_n = 1.0 if (toks1 and toks2 and s1_n.split()[0] == c_n.split()[0]) else 0.0
        
        X[i, 0] = r_ratio
        X[i, 1] = r_part
        X[i, 2] = r_sort
        X[i, 3] = r_set
        X[i, 4] = n_jac
        X[i, 5] = n_g3
        X[i, 6] = len_ratio
        X[i, 7] = exact_n
        X[i, 8] = first4_n
        X[i, 9] = first_tok_n
        
        # 2. Transliteration (Indic scripts)
        is_ascii = c_n.isascii()
        X[i, 10] = 0.0 if is_ascii else 1.0
        if not is_ascii:
            c_n_asc = to_ascii(c_n).lower()
            tr_ratio = fuzz.ratio(s1_n, c_n_asc) / 100.0
            tr_sort = fuzz.token_sort_ratio(s1_n, c_n_asc) / 100.0
            c_toks_asc = set(c_n_asc.replace('-', ' ').split())
            tr_un = toks1 | c_toks_asc
            tr_jac = (len(toks1 & c_toks_asc) / len(tr_un)) if tr_un else 0.0
            tr_g2 = get_char_3grams(c_n_asc)
            tr_gun = g1 | tr_g2
            tr_g3 = (len(g1 & tr_g2) / len(tr_gun)) if tr_gun else 0.0
        else:
            tr_ratio = r_ratio
            tr_sort = r_sort
            tr_jac = n_jac
            tr_g3 = n_g3
            
        X[i, 11] = tr_ratio
        X[i, 12] = tr_sort
        X[i, 13] = tr_jac
        X[i, 14] = tr_g3
        
        # 3. Domain / URL match
        c_dom = clean_domain(c_n)
        s1_concat = s1_n.replace(' ', '')
        c_dom_concat = c_dom.replace(' ', '')
        dom_exact = 1.0 if (s1_concat and s1_concat == c_dom_concat) else 0.0
        dom_contains = 1.0 if (len(c_dom_concat) >= 5 and c_dom_concat in s1_concat) or (len(s1_concat) >= 5 and s1_concat in c_dom_concat) else 0.0
        dom_ratio = fuzz.ratio(s1_concat, c_dom_concat) / 100.0
        X[i, 15] = dom_exact
        X[i, 16] = dom_contains
        X[i, 17] = dom_ratio
        
        # 4. Address features
        if s1_a and c_a:
            ar_ratio = fuzz.ratio(s1_a, c_a) / 100.0
            ar_sort = fuzz.token_sort_ratio(s1_a, c_a) / 100.0
            ar_set = fuzz.token_set_ratio(s1_a, c_a) / 100.0
            atoks1 = set(s1_a.split())
            atoks2 = set(c_a.split())
            aun = atoks1 | atoks2
            a_jac = (len(atoks1 & atoks2) / len(aun)) if aun else 0.0
            ag1 = get_char_3grams(s1_a)
            ag2 = get_char_3grams(c_a)
            agun = ag1 | ag2
            a_g3 = (len(ag1 & ag2) / len(agun)) if agun else 0.0
        else:
            ar_ratio = 0.0
            ar_sort = 0.0
            ar_set = 0.0
            a_jac = 0.0
            a_g3 = 0.0
            
        X[i, 18] = ar_ratio
        X[i, 19] = ar_sort
        X[i, 20] = ar_set
        X[i, 21] = a_jac
        X[i, 22] = a_g3
        X[i, 23] = 1.0 if not c_a else 0.0
        
        # 5. Numeric & Conflict suppression
        exact_k = 1.0 if (s1_k and c_k and s1_k == c_k) else 0.0
        nums1 = set(s1_k.split("|")) if s1_k else set()
        nums2 = set(c_k.split("|")) if c_k else set()
        num_un = nums1 | nums2
        k_jac = (len(nums1 & nums2) / len(num_un)) if num_un else 0.0
        k_diff = abs(len(nums1) - len(nums2))
        k_conflict = 1.0 if (nums1 and nums2 and not (nums1 & nums2)) else 0.0
        first_num_match = 1.0 if (s1_k and c_k and s1_k.split('|')[0] == c_k.split('|')[0]) else 0.0
        
        X[i, 24] = exact_k
        X[i, 25] = k_jac
        X[i, 26] = k_diff
        X[i, 27] = k_conflict
        X[i, 28] = first_num_match
        
        # 6. Composite & Interactions
        max_n_sim = max(r_ratio, tr_ratio, dom_ratio)
        max_n_jac = max(n_jac, tr_jac)
        high_addr_anchor = 1.0 if ((a_jac >= 0.50 or ar_ratio >= 0.65) and k_conflict == 0.0) else 0.0
        harmonic = (2.0 * max_n_sim * ar_ratio) / (max_n_sim + ar_ratio + 1e-5)
        
        X[i, 29] = max_n_sim
        X[i, 30] = max_n_jac
        X[i, 31] = high_addr_anchor
        X[i, 32] = harmonic
        
        # 7. Source
        X[i, 33] = 1.0 if c_src == "S2" else 0.0
        
    return X
