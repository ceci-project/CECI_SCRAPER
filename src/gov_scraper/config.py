"""Configuration constants for the Israeli Government Decisions Scraper."""

import os
from dotenv import load_dotenv

def find_env_file():
    """Find .env file in various locations for portability."""
    # Try multiple locations in order of preference
    locations = [
        os.path.join(os.getcwd(), '.env'),                    # Current working directory
        os.path.join(os.path.dirname(__file__), '.env'),      # Same directory as config.py
        os.path.join(os.path.dirname(__file__), '..', '.env'), # Parent directory (src/.env)
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'), # Project root
        os.path.join(os.path.expanduser('~'), '.env'),        # User home directory
    ]
    
    for location in locations:
        if os.path.exists(location):
            return location
    
    return None

# Load environment variables from the first found .env file
env_file = find_env_file()
if env_file:
    load_dotenv(env_file)
    print(f"Loaded environment variables from: {env_file}")
else:
    print("Warning: No .env file found. Please ensure environment variables are set.")

# URLs
BASE_CATALOG_URL = 'https://www.gov.il/he/collectors/policies'
CATALOG_PARAMS = {
    'Type': '30280ed5-306f-4f0b-a11d-cacf05d36648',
    'skip': 0,
    'limit': 5  # Start with 5 decisions
}
BASE_DECISION_URL = 'https://www.gov.il'

# HTTP Settings
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

# Fixed values for decisions (current government)
GOVERNMENT_NUMBER = 37
PRIME_MINISTER = "בנימין נתניהו"

# Prime Minister mapping by government number (governments 25-37)
PM_BY_GOVERNMENT = {
    25: "יצחק רבין",        # 1992-1995
    26: "שמעון פרס",         # 1995-1996
    27: "בנימין נתניהו",     # 1996-1999
    28: "אהוד ברק",          # 1999-2001
    29: "אריאל שרון",        # 2001-2003
    30: "אריאל שרון",        # 2003-2006
    31: "אהוד אולמרט",       # 2006-2009
    32: "בנימין נתניהו",     # 2009-2013
    33: "בנימין נתניהו",     # 2013-2015
    34: "בנימין נתניהו",     # 2015-2020
    35: "בנימין נתניהו",     # 2020-2021
    36: "נפתלי בנט",         # 2021-06-13 to 2022-07-01 (Bennett), then Lapid
    37: "בנימין נתניהו",     # 2022-present
}

# Government 36 rotation: Bennett until 2022-07-01, then Lapid
_GOV36_LAPID_START = "2022-07-01"


def get_pm_for_decision(gov_num: int, decision_date: str = None) -> str:
    """Get PM name for a decision, handling gov 36 rotation agreement.

    Args:
        gov_num: Government number (25-37)
        decision_date: ISO date string (YYYY-MM-DD), needed for gov 36

    Returns:
        PM name string
    """
    if gov_num == 36 and decision_date and decision_date >= _GOV36_LAPID_START:
        return "יאיר לפיד"
    return PM_BY_GOVERNMENT.get(gov_num, PRIME_MINISTER)

# Gemini Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

# AI Processing Configuration
USE_UNIFIED_AI = os.getenv('USE_UNIFIED_AI', 'true').lower() == 'true'
AI_CONFIDENCE_THRESHOLD = float(os.getenv('AI_CONFIDENCE_THRESHOLD', '0.5'))
AI_ENABLE_VALIDATION = os.getenv('AI_ENABLE_VALIDATION', 'true').lower() == 'true'

# Validate that Gemini API key is set - MANDATORY for operation
if not GEMINI_API_KEY:
    raise ValueError(
        "❌ GEMINI_API_KEY is not set!\n"
        "This system requires a valid Google Gemini API key to function.\n"
        "Please:\n"
        "  1. Copy .env.example to .env\n"
        "  2. Add your Gemini API key to the .env file\n"
        "  3. Get an API key from: https://aistudio.google.com/app/apikey"
    )

# Hebrew field labels for parsing
HEBREW_LABELS = {
    'date': 'תאריך פרסום:',
    'number': 'מספר החלטה:',
    'committee': 'ועדות שרים:'
}

# Output settings
OUTPUT_DIR = 'data'
OUTPUT_FILE = 'decisions_data.csv'
LOG_DIR = 'logs'
LOG_FILE = 'scraper.log'

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# CSV columns (excluding embedding, created_at, updated_at)
CSV_COLUMNS = [
    'id', 'decision_date', 'decision_number', 'committee', 'decision_title', 
    'decision_content', 'decision_url', 'summary', 'operativity', 
    'tags_policy_area', 'tags_government_body', 'tags_location', 'all_tags',
    'government_number', 'prime_minister', 'decision_key'
]