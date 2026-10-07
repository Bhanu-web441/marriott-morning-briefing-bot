import time
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Marriott Morning Briefing Bot",
    page_icon="🏨",
    layout="wide"
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_TITLE = "Marriott Morning Briefing Bot"

# Excel file must be in the same GitHub repository as app.py
FILE_NAME = "Marriott_Morning_Briefing_Bot.xlsx"

# Models are tried in this order.
# If the first model is temporarily unavailable, the app
# automatically attempts another model.
MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

MAX_RETRIES_PER_MODEL = 2


# ============================================================
# GET GEMINI API KEY
# ============================================================

def get_api_key():
    """
    Read the Gemini API key securely from Streamlit Secrets.
    Never place the real API key directly in this Python file.
    """

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


api_key = get_api_key()

if not api_key:
    st.error(
        "Gemini API key is not configured. "
        "Add GEMINI_API_KEY to Streamlit Secrets."
    )
    st.stop()


# Create Gemini client
client = genai.Client(api_key=api_key)


# ============================================================
# LOAD EXCEL DATA
# ============================================================

@st.cache_data
def load_data():
    """
    Load the simulated hotel operations workbook.

    The workbook contains:
    Occupancy
    Reservations
    VIP Requests
    HK Maintenance
    Staffing Roster
    Guest Reviews
    """

    file_path = Path(__file__).parent / FILE_NAME

    if not file_path.exists():
        raise FileNotFoundError(
            f"{FILE_NAME} was not found in the application folder."
        )

    occupancy = pd.read_excel(
        file_path,
        sheet_name="Occupancy",
        header=3
    )

    reservations = pd.read_excel(
        file_path,
        sheet_name="Reservations",
        header=3
    )

    vip = pd.read_excel(
        file_path,
        sheet_name="VIP Requests",
        header=3
    )

    maintenance = pd.read_excel(
        file_path,
        sheet_name="HK Maintenance",
        header=3
    )

    staffing = pd.read_excel(
        file_path,
        sheet_name="Staffing Roster",
        header=3
    )

    reviews = pd.read_excel(
        file_path,
        sheet_name="Guest Reviews",
        header=3
    )

    return (
        occupancy,
        reservations,
        vip,
        maintenance,
        staffing,
        reviews
    )


# ============================================================
# LOAD DATA SAFELY
# ============================================================

try:
    (
        occupancy,
        reservations,
        vip,
        maintenance,
        staffing,
        reviews
    ) = load_data()

except Exception as e:

    st.error("Unable to load the Marriott Excel dataset.")

    # Friendly diagnostic message without displaying a
    # complete Python traceback to public users.
    st.error(str(e))

    st.stop()


# ============================================================
# DATA CLEANING HELPER
# ============================================================

def clean_dataframe(df):
    """
    Remove completely empty rows and columns.
    """

    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    return df


occupancy = clean_dataframe(occupancy)
reservations = clean_dataframe(reservations)
vip = clean_dataframe(vip)
maintenance = clean_dataframe(maintenance)
staffing = clean_dataframe(staffing)
reviews = clean_dataframe(reviews)


# ============================================================
# CONVERT DATAFRAME TO TEXT
# ============================================================

def dataframe_to_text(df, max_rows=60):
    """
    Convert a dataframe into compact text for Gemini.

    Limiting rows helps control API usage and prompt size.
    """

    if df.empty:
        return "No records available."

    safe_df = df.head(max_rows).copy()

    # Convert missing values into readable blanks
    safe_df = safe_df.fillna("")

    return safe_df.to_string(index=False)


# ============================================================
# CREATE HOTEL DATA CONTEXT
# ============================================================

def build_hotel_context():
    """
    Create the operational information that Gemini can analyze.
    """

    context = f"""
SIMULATED HOTEL OPERATIONS DATA

IMPORTANT:
This is simulated Marriott case-study data used only for
academic purposes. It is not actual Marriott International
operational information.


==============================
OCCUPANCY
==============================

{dataframe_to_text(occupancy)}


==============================
RESERVATIONS
==============================

{dataframe_to_text(reservations)}


==============================
VIP REQUESTS
==============================

{dataframe_to_text(vip)}


==============================
HOUSEKEEPING / MAINTENANCE
==============================

{dataframe_to_text(maintenance)}


==============================
STAFFING
==============================

{dataframe_to_text(staffing)}


==============================
GUEST REVIEWS
==============================

{dataframe_to_text(reviews)}

"""

    return context


hotel_context = build_hotel_context()


# ============================================================
# GEMINI ERROR CLASSIFICATION
# ============================================================

def is_temporary_error(error):
    """
    Determine whether an API failure is likely temporary.
    """

    error_text = str(error).lower()

    temporary_terms = [
        "503",
        "unavailable",
        "high demand",
        "temporarily",
        "timeout",
        "timed out",
        "429",
        "resource_exhausted",
        "rate limit",
        "overloaded"
    ]

    return any(term in error_text for term in temporary_terms)


# ============================================================
# ASK MARRIOTT BOT
# ============================================================

def ask_marriott_bot(question):
    """
    Send the user's question and simulated hotel data to Gemini.

    Features:
    - retries temporary failures
    - tries backup models
    - avoids displaying technical traceback information
    """

    prompt = f"""
You are the AI assistant for an academic project called
Marriott Morning Briefing Bot.

Your purpose is to help hotel managers understand simulated
daily hotel operations data.

The records supplied below are simulated academic records.
They are NOT actual Marriott International operational data.

Answer the user's question using ONLY the supplied dataset.

Do not invent:
- guests
- reservation numbers
- occupancy percentages
- maintenance issues
- staffing shortages
- VIP requests
- review information
- dates
- operational events

If the requested information cannot be determined from the
dataset, clearly say that the available simulated data does
not provide enough information.

When appropriate:

1. Summarize the operational situation.
2. Identify important risks or exceptions.
3. Recommend practical management actions.
4. Prioritize urgent issues.
5. Keep the answer concise and useful for a hotel morning
   management briefing.

HOTEL DATA:

{hotel_context}


USER QUESTION:

{question}


Provide the management briefing answer:
"""

    last_error = None

    for model_name in MODEL_CANDIDATES:

        for attempt in range(MAX_RETRIES_PER_MODEL):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if response and response.text:
                    return response.text

            except Exception as e:

                last_error = e

                if is_temporary_error(e):

                    # Wait briefly before retrying.
                    time.sleep(2 + attempt)

                    continue

                # If the model is unavailable/not supported,
                # continue to the next candidate.
                error_text = str(e).lower()

                if (
                    "404" in error_text
                    or "not found" in error_text
                    or "model" in error_text
                ):
                    break

                return (
                    "I encountered a problem while analyzing "
                    "the hotel data. Please try again."
                )

    # Every model/retry failed
    return (
        "The AI service is temporarily busy or unavailable. "
        "Your hotel data loaded successfully, but the AI model "
        "could not respond right now. Please wait a few seconds "
        "and try your question again."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content":
                "Welcome to the Marriott Morning Briefing Bot. "
                "Ask me about hotel operations."
        }
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This chatbot analyzes a simulated 30-day "
        "Marriott hotel operations dataset."
    )

    st.write("**Data analyzed:**")

    st.write("• Occupancy")
    st.write("• Reservations")
    st.write("• VIP requests")
    st.write("• Maintenance")
    st.write("• Staffing")
    st.write("• Guest reviews")

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content":
                    "Welcome to the Marriott Morning Briefing Bot. "
                    "Ask me about hotel operations."
            }
        ]

        st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title("🏨 Marriott Morning Briefing Bot")

st.caption(
    "AI-assisted hotel operations briefing | "
    "Academic project using simulated Marriott data"
)

st.warning(
    "All hotel records used in this application are simulated "
    "for academic purposes and are not actual Marriott "
    "International operational data."
)


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.header("Suggested Questions")


suggested_questions = {
    "📊 Occupancy Situation":
        "What is the occupancy situation and which dates "
        "require the most management attention?",

    "🔧 Maintenance Priorities":
        "What maintenance issues should management prioritize?",

    "🧳 Arrivals and Reservations":
        "Summarize the most important upcoming arrivals "
        "and reservation activity.",

    "⭐ Pending VIP Requests":
        "Which VIP or special requests require attention?",

    "👥 Staffing Shortages":
        "Where are the most important staffing shortages "
        "or coverage gaps?",

    "🚨 Top 3 Management Actions":
        "What are the three most important actions "
        "management should take today?"
}


# ============================================================
# FUNCTION FOR HANDLING QUESTIONS
# ============================================================

def process_question(question):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.spinner("Analyzing hotel operations..."):

        answer = ask_marriott_bot(question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# QUESTION BUTTONS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "📊 Occupancy Situation",
        use_container_width=True
    ):
        process_question(
            suggested_questions["📊 Occupancy Situation"]
        )
        st.rerun()

    if st.button(
        "⭐ Pending VIP Requests",
        use_container_width=True
    ):
        process_question(
            suggested_questions["⭐ Pending VIP Requests"]
        )
        st.rerun()


with col2:

    if st.button(
        "🔧 Maintenance Priorities",
        use_container_width=True
    ):
        process_question(
            suggested_questions["🔧 Maintenance Priorities"]
        )
        st.rerun()

    if st.button(
        "👥 Staffing Shortages",
        use_container_width=True
    ):
        process_question(
            suggested_questions["👥 Staffing Shortages"]
        )
        st.rerun()


with col3:

    if st.button(
        "🧳 Arrivals and Reservations",
        use_container_width=True
    ):
        process_question(
            suggested_questions["🧳 Arrivals and Reservations"]
        )
        st.rerun()

    if st.button(
        "🚨 Top 3 Management Actions",
        use_container_width=True
    ):
        process_question(
            suggested_questions["🚨 Top 3 Management Actions"]
        )
        st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask about hotel operations..."
)


if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Display the user's question immediately
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing hotel operations..."
        ):

            answer = ask_marriott_bot(
                user_question
            )

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
