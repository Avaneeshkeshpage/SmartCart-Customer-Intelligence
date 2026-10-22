"""Action guidance aligned to the behavior-first SmartCart segments."""

MARKETING_ACTIONS = {
    "High-Value Digital Active": {
        "description": "High spending and frequent purchases, with a relatively strong online purchase share.",
        "action": "Test loyalty benefits, online cross-category recommendations, and early access; measure incremental lift.",
    },
    "High-Value Store/Catalog Active": {
        "description": "High spending and frequent purchases, with comparatively greater store/catalog activity.",
        "action": "Test curated catalog bundles and store-linked loyalty benefits; measure incremental lift.",
    },
    "Deal-Driven Regulars": {
        "description": "Purchase activity is relatively frequent and deal purchases are comparatively prominent.",
        "action": "Test targeted promotions with margin controls and compare response against a holdout group.",
    },
    "Digital-First Researchers": {
        "description": "Frequent web visits and relatively strong digital engagement, with lower spending than high-value groups.",
        "action": "Improve product discovery, recommend affordable complementary products, and test relevant online offers.",
    },
    "High-Spend Occasional": {
        "description": "High spending with comparatively fewer recorded purchases.",
        "action": "Use personalized replenishment reminders and relevant product bundles; avoid assuming the customer is loyal without repeat evidence.",
    },
    "Low-Engagement": {
        "description": "Lower observed purchase activity and/or spending in the available features.",
        "action": "Use low-cost re-engagement experiments and review channel friction before increasing campaign spend.",
    },
    "At-Risk / Inactive": {
        "description": "Longer time since the last purchase relative to the other clusters.",
        "action": "Run a measured win-back experiment with a relevant reminder or incentive, and compare against a holdout.",
    },
}

def get_action(segment_name: str) -> dict:
    return MARKETING_ACTIONS.get(
        segment_name,
        {"description": "Behavioral profile requires review.", "action": "Inspect the segment metrics before selecting a campaign."}
    )

NEW_CUSTOMER_ACTIONS = {
    "new_customer": {
        "description": "An existing dataset record meets the current cold-start heuristic.",
        "action": "Use onboarding content and a first/second-purchase incentive; reassess after behavior is observed.",
    },
    "unknown_id": {
        "description": "The entered ID is not present in the current dataset.",
        "action": "For a genuinely new customer, enter their observed behavior under Predict New Customer or use onboarding until data is available.",
    },
}

def get_new_customer_action(status: str) -> dict:
    return NEW_CUSTOMER_ACTIONS.get(
        status, {"description": "Unknown status.", "action": "Review manually."}
    )
