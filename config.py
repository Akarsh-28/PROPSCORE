"""
All file paths and constants live here.
If you rename a file, this is the only place you should need to change it.
"""

#  INPUT 
# Path to your raw data file. Change this to wherever r2c_data.csv actually lives.
INPUT_CSV = "r2c_data.csv"

#  OUTPUTS 
SCORED_USERS_CSV = "r2c_scored_user.csv"
FEATURE_IMPORTANCE_CSV = "r2c_feature_importance.csv"

#  COLUMNS 
TARGET_COL = "converted_to_paid"

# Columns only known AFTER a user converts -- must be excluded from features,
# otherwise the model "cheats" by seeing the answer.
LEAKAGE_COLS = [
    "plan_purchased",
    "payment_amount_inr",
    "days_to_conversion",
    "payment_date",
    "churned",
]

# Identifier / raw-date columns not useful as model features directly.
ID_COLS = [
    "user_id",
    "institution_or_company",
    "signup_date",
    "first_visit_date",
    "last_active_date",
    TARGET_COL,
]

# Feature-usage flags that are blank for some user_types (e.g. industry-only
# features are blank for researcher rows) -- these get filled with 0.
BOOL_FLAG_COLS = [
    "used_crs_score",
    "used_gap_analysis",
    "used_industry_fit",
    "used_comparison_tool",
    "used_pitch_export",
    "used_due_diligence",
    "used_ip_patent_check",
]

# Text columns to one-hot encode.
CATEGORICAL_COLS = ["user_type", "country", "traffic_source", "device_type"]

#  MODEL SETTINGS 
RANDOM_STATE = 42
N_CV_FOLDS = 5
TOP_K_FRACTION = 0.20  # "top 20% of users by score" for the lead-list metric

MODEL_PARAMS = dict(
    max_iter=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE,
)