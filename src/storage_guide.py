"""Food Storage & Eat-By Knowledge Base for Food Freshness Detection.

Provides realistic, scientifically-informed produce storage recommendations,
estimated shelf life, eat-by timelines, and food-safety guidance based on
detected food category and classified freshness condition.
"""

from typing import Dict, Any

STORAGE_GUIDE_DB: Dict[str, Dict[str, Any]] = {
    "Banana": {
        "recommended_storage": "Room temperature (cool, dry place away from direct sunlight; avoid refrigeration to prevent peel browning)",
        "typical_storage_life": "5 – 7 days from harvest / purchase",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 3 – 5 days for optimal firmness and natural sweetness",
                "estimated_remaining": "4 – 6 days",
                "recommendation": "Store loose on a countertop or hanging hook with good ventilation. Keep stems wrapped or separated from other produce to prevent accelerated ripening.",
            },
            "moderate": {
                "freshness_display": "Moderate (Ripe)",
                "best_to_eat": "Within 24 – 48 hours (sugar spots indicate peak sweetness)",
                "estimated_remaining": "1 – 2 days",
                "recommendation": "Brown sugar spots indicate peak sugar content. Ideal for smoothies, oatmeal, baking banana bread, or peeling and freezing in airtight containers.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard or compost safely. Completely blackened, leaky, or moldy bananas may harbor mycotoxins and are unsafe for consumption.",
            },
        },
    },
    "Bittermelon": {
        "recommended_storage": "Refrigerator crisper drawer (loosely wrapped in a dry paper towel inside a perforated bag)",
        "typical_storage_life": "4 – 6 days refrigerated",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 3 – 4 days to preserve firm texture and crisp bitterness",
                "estimated_remaining": "3 – 5 days",
                "recommendation": "Keep surface completely dry; excess moisture causes rapid softening. Do not wash until immediately before cooking.",
            },
            "moderate": {
                "freshness_display": "Moderate (Ripening)",
                "best_to_eat": "Within 24 – 36 hours (seeds and skin turning yellow/orange)",
                "estimated_remaining": "1 – 2 days",
                "recommendation": "As bittermelon ripens, seeds turn red and flesh softens. Cook thoroughly in stir-fries, curries, or teas promptly before rot begins.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard safely. Soft watery depressions, mold colonies, or sour fermenting odor indicate bacterial breakdown.",
            },
        },
    },
    "Cucumber": {
        "recommended_storage": "Refrigerator crisper drawer (wrapped in dry paper towel; avoid coldest back of fridge)",
        "typical_storage_life": "7 – 10 days refrigerated",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 4 – 6 days for maximum hydration, crispness, and flavor",
                "estimated_remaining": "5 – 7 days",
                "recommendation": "Keep dry in the vegetable crisper. Store away from ethylene-releasing fruits like apples and bananas which cause rapid yellowing.",
            },
            "moderate": {
                "freshness_display": "Moderate",
                "best_to_eat": "Within 1 – 2 days (skin dulling or tips softening slightly)",
                "estimated_remaining": "2 – 3 days",
                "recommendation": "Peel outer skin if slightly bitter; use quickly in sliced salads, tzatziki, cold gazpacho, or infused water.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard immediately. Sunken watery spots, skin slippage, or slimy surface indicate bacterial breakdown.",
            },
        },
    },
    "Eggplant": {
        "recommended_storage": "Cool pantry or vegetable crisper (45–50°F / 7–10°C; avoid prolonged freezing temperatures)",
        "typical_storage_life": "4 – 7 days",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 3 – 5 days while skin is smooth, glossy, and firm",
                "estimated_remaining": "4 – 6 days",
                "recommendation": "Eggplants are chill-sensitive. Store unwashed in a loosely closed paper bag or front section of the crisper drawer.",
            },
            "moderate": {
                "freshness_display": "Moderate",
                "best_to_eat": "Within 24 – 48 hours (skin losing luster or slight wrinkles appearing)",
                "estimated_remaining": "1 – 2 days",
                "recommendation": "Check interior upon cutting. If flesh is still pale, cook thoroughly by roasting, grilling, or stewing in curries/baba ganoush.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard safely. Brown spongy flesh, soft sunken spots, or foul odor indicate cellular breakdown and mold.",
            },
        },
    },
    "Orange": {
        "recommended_storage": "Refrigerator crisper drawer for long term (or cool room temperature for up to 1 week)",
        "typical_storage_life": "2 – 3 weeks refrigerated (1 week at room temp)",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 10 – 14 days refrigerated (3 – 5 days at room temperature)",
                "estimated_remaining": "10 – 14 days",
                "recommendation": "Store loose in the crisper drawer with good airflow. Avoid sealed plastic bags which trap moisture and encourage penicillium mold.",
            },
            "moderate": {
                "freshness_display": "Moderate",
                "best_to_eat": "Within 2 – 4 days (skin beginning to dry or soften)",
                "estimated_remaining": "3 – 5 days",
                "recommendation": "Juice immediately or zest peel for culinary use before segments lose moisture and aromatic oils.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard immediately. Blue-green powdery mold (Penicillium digitatum) or squishy rotting spots spread quickly to other citrus.",
            },
        },
    },
    "Papaya": {
        "recommended_storage": "Room temperature until yellow-orange, then Refrigerator crisper drawer",
        "typical_storage_life": "5 – 7 days once ripe (refrigerated)",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 3 – 5 days when skin yields gently to light thumb pressure",
                "estimated_remaining": "4 – 6 days",
                "recommendation": "Once ripe, store in the refrigerator loosely wrapped in perforated plastic. Halve, deseed, and serve with a squeeze of fresh lime.",
            },
            "moderate": {
                "freshness_display": "Moderate (Very Ripe)",
                "best_to_eat": "Within 24 – 36 hours (skin soft and highly fragrant)",
                "estimated_remaining": "1 – 2 days",
                "recommendation": "Flesh will be very tender. Scoop and puree into tropical smoothies, breakfast parfaits, or meat tenderizing marinades.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard safely. Dark sunken patches, internal fungal growth, or fermented alcohol smell indicate bacterial spoilage.",
            },
        },
    },
    "Pineapple": {
        "recommended_storage": "Refrigerator crisper drawer (whole) or sealed container (cut chunks)",
        "typical_storage_life": "5 – 7 days refrigerated (whole) / 3 – 4 days (cut)",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 3 – 5 days for optimal sweetness and tropical acidity",
                "estimated_remaining": "4 – 6 days",
                "recommendation": "Pineapples do not ripen further after harvest. Store whole in the refrigerator, or cut and refrigerate in an airtight glass container.",
            },
            "moderate": {
                "freshness_display": "Moderate (Ripe)",
                "best_to_eat": "Within 24 – 48 hours (deep gold skin, strong sweet fragrance)",
                "estimated_remaining": "1 – 2 days",
                "recommendation": "Cut and freeze chunks for frozen desserts, caramelize on a grill, or incorporate into stir-fries and sweet sauces.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard. A pungent vinegar or alcoholic odor, leaking soggy base, or brown soft pulp indicates fermentation and rot.",
            },
        },
    },
    "Tomato": {
        "recommended_storage": "Room temperature stem-side down (avoid refrigeration to preserve aromatic flavor volatiles)",
        "typical_storage_life": "5 – 7 days at room temperature",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 3 – 5 days at peak red ripeness",
                "estimated_remaining": "4 – 7 days",
                "recommendation": "Store stem-side down at room temperature out of direct sunlight. Chilling below 55°F (12°C) irreversibly degrades flavor enzymes.",
            },
            "moderate": {
                "freshness_display": "Moderate (Softening)",
                "best_to_eat": "Within 24 – 48 hours (deep color, yielding skin)",
                "estimated_remaining": "1 – 2 days",
                "recommendation": "Soft tomatoes are ideal for homemade marinara sauces, tomato soup, shakshuka, roasting, or canning.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard immediately. Black mold at the stem, leaking watery lesions, or sour rotting smells signify hazardous spoilage.",
            },
        },
    },
    "Apple": {
        "recommended_storage": "Refrigerator",
        "typical_storage_life": "2 – 4 weeks",
        "conditions": {
            "fresh": {
                "freshness_display": "Fresh",
                "best_to_eat": "Within 3 – 5 days",
                "estimated_remaining": "3 – 5 days",
                "recommendation": "Looks fresh and healthy. Keep refrigerated and consume within a few days for the best taste and nutrition.",
            },
            "moderate": {
                "freshness_display": "Moderate",
                "best_to_eat": "Within 2 – 4 days",
                "estimated_remaining": "2 – 3 days",
                "recommendation": "Produce is softening. Great for applesauce, baking, or stewing.",
            },
            "spoiled": {
                "freshness_display": "Spoiled",
                "best_to_eat": "DO NOT EAT (Do not consume)",
                "estimated_remaining": "0 days",
                "recommendation": "Discard safely. Soft rot, deep browning, or mold make it unsafe for consumption.",
            },
        },
    },
}

# Generic fallback for produce items not in database
DEFAULT_STORAGE_GUIDE = {
    "recommended_storage": "Cool, dry pantry or refrigerator crisper drawer",
    "typical_storage_life": "4 – 7 days typical storage",
    "conditions": {
        "fresh": {
            "freshness_display": "Fresh",
            "best_to_eat": "Within 3 – 5 days for optimal quality",
            "estimated_remaining": "3 – 5 days",
            "recommendation": "Store in a well-ventilated container in your refrigerator crisper or cool pantry. Check regularly for moisture buildup.",
        },
        "moderate": {
            "freshness_display": "Moderate",
            "best_to_eat": "Within 24 – 48 hours",
            "estimated_remaining": "1 – 2 days",
            "recommendation": "Produce is reaching maturity. Cook thoroughly in soups, stir-fries, or baked recipes promptly.",
        },
        "spoiled": {
            "freshness_display": "Spoiled",
            "best_to_eat": "DO NOT EAT (Do not consume)",
            "estimated_remaining": "0 days",
            "recommendation": "Discard safely. Visual signs of degradation, mold, or odor make it unsafe for consumption.",
        },
    },
}


def get_food_storage_info(food_name: str, freshness_code: str) -> Dict[str, str]:
    """Retrieves scientifically sound storage and eat-by guidance for detected produce.

    Args:
        food_name: The detected produce name (e.g. 'Orange', 'Tomato', 'Banana').
        freshness_code: Normalized freshness code ('fresh', 'moderate', or 'spoiled').

    Returns:
        Dict with recommended storage, storage life, eat-by timeframe, remaining days,
        actionable recommendation, and food safety advisory notice.
    """
    clean_food = food_name.strip().title()
    clean_code = freshness_code.strip().lower()

    if clean_code not in ("fresh", "moderate", "spoiled"):
        clean_code = "fresh"

    data = STORAGE_GUIDE_DB.get(clean_food, DEFAULT_STORAGE_GUIDE)
    cond_data = data["conditions"].get(clean_code, data["conditions"]["fresh"])

    return {
        "food": clean_food,
        "current_freshness": cond_data["freshness_display"],
        "recommended_storage": data["recommended_storage"],
        "estimated_storage_life": data["typical_storage_life"],
        "best_to_eat": cond_data["best_to_eat"],
        "estimated_remaining_time": cond_data["estimated_remaining"],
        "recommendation": cond_data["recommendation"],
        "food_safety_advisory": (
            "Storage timelines and eat-by dates are typical estimates based on visual surface quality. "
            "Optical AI inspection cannot detect internal pathogens or guarantee food safety. "
            "Always inspect produce for unusual odor, texture, slime, or mold before consumption."
        ),
    }
