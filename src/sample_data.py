"""
Curated real-world benchmark test cases illustrating:
1. Alias + Address Anchor Matching
2. Indic Script Transliteration (Tamil, Devanagari)
3. Domain / URL Normalization
4. False-Positive Distractor Suppression
5. True Singleton Credit
"""

PRESET_CASES = {
    "🇺🇸 US: Alias & Address Anchor (Maure Williams Colombier)": {
        "query": {
            "entity_id": "S1-965667",
            "business_name": "Maure Williams Colombier Inc",
            "business_address": "85 Wayne Avenue, Ticonderoga, NY 12883",
            "country": "US"
        },
        "description": "Demonstrates synthetic alias replacement (Dréxkor) rescued by physical address anchor, URL matching, and name-only typo handling.",
        "candidates": [
            {
                "entity_id": "S2-681193310",
                "source": "S2",
                "business_name": "Maure Wilblims Colombier Inc",
                "business_address": "",
                "is_true_match": True,
                "notes": "Typo in name ('Wilblims'), address missing/NaN in Source 2."
            },
            {
                "entity_id": "S2-743505751",
                "source": "S2",
                "business_name": "Maure Williams Colombier",
                "business_address": "",
                "is_true_match": True,
                "notes": "Dropped 'Inc' corporate suffix, address missing."
            },
            {
                "entity_id": "S3-775321672",
                "source": "S3",
                "business_name": "Dréxkor",
                "business_address": "85 Wanye Avenue, Ticonderoga Townshiip, New York",
                "is_true_match": True,
                "notes": "Synthetic fantasy alias ('Dréxkor') anchored by identical building number & street."
            },
            {
                "entity_id": "S3-11291185",
                "source": "S3",
                "business_name": "maurewilliamscolombier.com",
                "business_address": "Wayne Ave, Ticonderoga Townshiip, New York",
                "is_true_match": True,
                "notes": "Domain URL format with stripped TLD (.com)."
            },
            {
                "entity_id": "S2-999900111",
                "source": "S2",
                "business_name": "Williams Financial Group LLC",
                "business_address": "85 Wayne Avenue, Jersey City, NJ 07306",
                "is_true_match": False,
                "notes": "Hard Negative: Shares partial name and street number, but different company & city."
            }
        ]
    },
    "🇮🇳 India: Multilingual Indic Transliteration (Raj Investments)": {
        "query": {
            "entity_id": "S1-55344266",
            "business_name": "Raj Investments LLP",
            "business_address": "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu 600004",
            "country": "India"
        },
        "description": "Demonstrates cross-script matching across Tamil and Devanagari scripts converted via AnyAscii phonetics.",
        "candidates": [
            {
                "entity_id": "S2-249013014",
                "source": "S2",
                "business_name": "ராஜ் இன்வெஸ்ட்மென்ட்ஸ்",
                "business_address": "6/29 CIT Colony 2nd Main Rd Mylapore Chennai",
                "is_true_match": True,
                "notes": "Tamil Unicode script transliterates phonetically to 'raj invessttments'."
            },
            {
                "entity_id": "S3-478195123",
                "source": "S3",
                "business_name": "राज इन्वेस्टमेंट्स एलएलपी",
                "business_address": "6 29 CIT Colony Mylapore Chennai Tamil Nadu 600004",
                "is_true_match": True,
                "notes": "Devanagari script transliterates phonetically to 'raj investtments elllpi'."
            },
            {
                "entity_id": "S2-981240192",
                "source": "S2",
                "business_name": "Rajesh Investment Services",
                "business_address": "12 C.I.T. Nagar, Nandanam, Chennai 600035",
                "is_true_match": False,
                "notes": "Hard Negative: Similar name prefix and Chennai location, but different entity."
            }
        ]
    },
    "🇫🇷 France: Accent Variations & Street Abbreviations": {
        "query": {
            "entity_id": "S1-102811957",
            "business_name": "Société Générale de Distribution Industrielle",
            "business_address": "14 Boulevard Haussmann, 75009 Paris",
            "country": "France"
        },
        "description": "Demonstrates French diacritics normalization (é, è) and common European street abbreviations (Bd -> Boulevard).",
        "candidates": [
            {
                "entity_id": "S2-478959098",
                "source": "S2",
                "business_name": "Soc Gen de Distribution Industrielle",
                "business_address": "14 Bd Haussmann 75009 Paris",
                "is_true_match": True,
                "notes": "Abbreviated corporate name ('Soc Gen') and street ('Bd' -> 'Boulevard')."
            },
            {
                "entity_id": "S3-728090388",
                "source": "S3",
                "business_name": "sgdistribution.fr",
                "business_address": "14 Boulevard Haussmann, 75009 Paris",
                "is_true_match": True,
                "notes": "French country-code domain (.fr) matching physical headquarters address."
            },
            {
                "entity_id": "S3-881923019",
                "source": "S3",
                "business_name": "Société Nouvelle de Distribution",
                "business_address": "14 Boulevard Saint-Germain, 75005 Paris",
                "is_true_match": False,
                "notes": "Hard Negative: Similar French corporate words, but different street and arrondissement."
            }
        ]
    },
    "🛡️ True Singleton: Suppressing False Positives": {
        "query": {
            "entity_id": "S1-99882211",
            "business_name": "Quantum Horizon Biotech Labs Inc",
            "business_address": "402 Silicon Way, Suite 300, Austin, TX 78701",
            "country": "US"
        },
        "description": "Demonstrates high-precision suppression: When no true match exists, the model correctly predicts SINGLETON to earn full credit.",
        "candidates": [
            {
                "entity_id": "S2-441029481",
                "source": "S2",
                "business_name": "Quantum Dynamics Inc",
                "business_address": "1200 Tech Ridge Blvd, Austin, TX 78753",
                "is_true_match": False,
                "notes": "Shares the word 'Quantum' and city 'Austin', but numeric key conflicts (402 vs 1200)."
            },
            {
                "entity_id": "S3-559102948",
                "source": "S3",
                "business_name": "Horizon Bio Solutions LLC",
                "business_address": "500 Research Blvd, Dallas, TX 75201",
                "is_true_match": False,
                "notes": "Shares 'Horizon Bio', but located in Dallas with conflicting address."
            }
        ]
    }
}
