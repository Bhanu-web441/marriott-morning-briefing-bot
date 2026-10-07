import pandas as pd
import streamlit as st
from pathlib import Path
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
# SETTINGS
# ============================================================

FILE_NAME = "Marriott_Morning_Briefing_Bot.xlsx"

# Try these models in order.
# If one is temporarily unavailable, move immediately
# to the next model instead of waiting through many retries.
MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview"
]


# ============================================================
# GEMINI API KEY
# ============================================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None


if not api_key:
    st.error(
        "Gemini API key is not configured. "
        "Add GEMINI_API_KEY to Streamlit Secrets."
    )
    st.stop()


client = genai.Client(api_key=api_key)


# ============================================================
# LOAD EXCEL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    file_path = Path(__file__).parent / FILE_NAME

    if not file_path.exists():
        raise FileNotFoundError(
            f"{FILE_NAME} was not found."
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

    return {
        "occupancy": occupancy,
        "reservations": reservations,
        "vip": vip,
        "maintenance": maintenance,
        "staffing": staffing,
        "reviews": reviews
    }


# ============================================================
# LOAD DATA
# ============================================================

try:
    datasets = load_data()

except Exception as e:

    st.error(
        "Unable to load the simulated hotel dataset."
    )

    st.error(str(e))

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataframe(df):

    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    return df


for key in datasets:
    datasets[key] = clean_dataframe(
        datasets[key]
    )


# ============================================================
# DATAFRAME TO TEXT
# ============================================================

def dataframe_to_text(df, max_rows=40):
    """
    Convert only the necessary portion of a dataframe
    into text for Gemini.

    Limiting rows makes requests smaller and faster.
    """

    if df.empty:
        return "No records available."

    data = df.head(max_rows).copy()

    data = data.fillna("")

    return data.to_string(
        index=False
    )


# ============================================================
# QUESTION ROUTER
# ============================================================

def determine_relevant_data(question):
    """
    Determine which hotel datasets are relevant.

    This prevents sending the complete workbook to Gemini
    for simple questions.
    """

    q = question.lower()

    selected = []


    # OCCUPANCY
    occupancy_words = [
        "occupancy",
        "occupied",
        "rooms booked",
        "available rooms",
        "room availability",
        "demand",
        "high demand"
    ]

    if any(word in q for word in occupancy_words):
        selected.append("occupancy")


    # RESERVATIONS
    reservation_words = [
        "reservation",
        "reservations",
        "arrival",
        "arrivals",
        "booking",
        "bookings",
        "check-in",
        "check in",
        "check-out",
        "check out"
    ]

    if any(word in q for word in reservation_words):
        selected.append("reservations")


    # VIP REQUESTS
    vip_words = [
        "vip",
        "special request",
        "special requests",
        "guest request",
        "guest requests",
        "pending request"
    ]

    if any(word in q for word in vip_words):
        selected.append("vip")


    # MAINTENANCE
    maintenance_words = [
        "maintenance",
        "repair",
        "repairs",
        "issue",
        "issues",
        "housekeeping",
        "broken",
        "fault",
        "critical"
    ]

    if any(word in q for word in maintenance_words):
        selected.append("maintenance")


    # STAFFING
    staffing_words = [
        "staff",
        "staffing",
        "shortage",
        "shortages",
        "coverage",
        "employee",
        "employees",
        "roster",
        "shift",
        "shifts"
    ]

    if any(word in q for word in staffing_words):
        selected.append("staffing")


    # REVIEWS
    review_words = [
        "review",
        "reviews",
        "rating",
        "ratings",
        "feedback",
        "complaint",
        "complaints",
        "guest satisfaction"
    ]

    if any(word in q for word in review_words):
        selected.append("reviews")


    # MANAGEMENT / EXECUTIVE QUESTIONS
    # These require information across several datasets.

    management_words = [
        "management",
        "manager",
        "morning briefing",
        "executive briefing",
        "top three",
        "top 3",
        "priorities",
        "priority",
        "important actions",
        "operational risk",
        "operational risks",
        "overall situation",
        "hotel situation"
    ]

    if any(word in q for word in management_words):

        selected = [
            "occupancy",
            "reservations",
            "vip",
            "maintenance",
            "staffing",
            "reviews"
        ]


    # If no category is recognized,
    # send a limited version of all datasets.

    if not selected:

        selected = [
            "occupancy",
            "reservations",
            "vip",
            "maintenance",
            "staffing",
            "reviews"
        ]


    # Remove duplicates while maintaining order

    selected = list(
        dict.fromkeys(selected)
    )

    return selected


# ============================================================
# BUILD ONLY NECESSARY CONTEXT
# ============================================================

def build_context(question):

    relevant = determine_relevant_data(
        question
    )

    sections = []


    labels = {

        "occupancy":
            "OCCUPANCY",

        "reservations":
            "RESERVATIONS",

        "vip":
            "VIP AND SPECIAL REQUESTS",

        "maintenance":
            "HOUSEKEEPING AND MAINTENANCE",

        "staffing":
            "STAFFING ROSTER",

        "reviews":
            "GUEST REVIEWS"
    }


    for key in relevant:

        sections.append(
            f"""
==============================
{labels[key]}
==============================

{dataframe_to_text(datasets[key])}
"""
        )


    return "\n".join(sections)


# ============================================================
# CHECK TEMPORARY API ERRORS
# ============================================================

def temporary_api_error(error):

    text = str(error).lower()

    temporary_terms = [
        "503",
        "unavailable",
        "high demand",
        "overloaded",
        "429",
        "resource_exhausted",
        "rate limit",
        "timeout",
        "timed out"
    ]

    return any(
        term in text
        for term in temporary_terms
    )


# ============================================================
# ASK GEMINI
# ============================================================

def ask_marriott_bot(question):

    context = build_context(
        question
    )

    prompt = f"""
You are the Marriott Morning Briefing Bot,
an AI assistant created for an academic hotel
operations project.

IMPORTANT DATA RULE:

The information below is simulated academic
Marriott case-study data.

It is NOT actual Marriott International
operational data.

Answer the user's question using ONLY the
provided simulated hotel records.

Do not invent information.

Do not invent:

guest names,
reservation IDs,
occupancy percentages,
dates,
maintenance problems,
VIP requests,
staffing shortages,
ratings,
or operational events.

If the dataset does not contain enough
information to answer the question, clearly
state:

"The available simulated dataset does not
provide enough information to determine that."

For management questions:

1. Identify the most important findings.
2. Prioritize urgent operational problems.
3. Explain why they matter.
4. Recommend practical management actions.

Keep normal answers concise.

Use bullet points when they improve readability.

Do not unnecessarily repeat the academic
disclaimer in every answer.


SIMULATED HOTEL DATA:

{context}


USER QUESTION:

{question}


ANSWER:
"""


    # ========================================================
    # FAST MODEL FALLBACK
    # ========================================================

    for model_name in MODEL_CANDIDATES:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response and response.text:

                return response.text


        except Exception as e:

            error_text = str(e).lower()

            # Immediately try next model for temporary errors.
            # No long sleep/retry cycle.

            if temporary_api_error(e):
                continue

            # Also move to backup if model is unavailable.

            if (
                "404" in error_text
                or "not found" in error_text
                or "model" in error_text
            ):
                continue

            return (
                "I encountered a temporary problem while "
                "analyzing the hotel data. Please try again."
            )


    return (
        "The AI service is currently busy. "
        "The hotel dataset loaded successfully, but an AI "
        "response could not be generated right now. "
        "Please try again in a few moments."
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
        "This chatbot analyzes a simulated "
        "30-day Marriott hotel operations dataset."
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
                    "Welcome to the Marriott Morning "
                    "Briefing Bot. Ask me about "
                    "hotel operations."
            }

        ]

        st.rerun()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏨 Marriott Morning Briefing Bot"
)

st.caption(
    "AI-assisted hotel operations briefing | "
    "Academic project using simulated Marriott data"
)


st.warning(
    "All hotel records used in this application "
    "are simulated for academic purposes and are "
    "not actual Marriott International operational data."
)


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.header(
    "Suggested Questions"
)


QUESTIONS = {

    "occupancy":
        "What is the occupancy situation and which "
        "dates require the most management attention?",

    "maintenance":
        "What maintenance issues should management "
        "prioritize?",

    "reservations":
        "Summarize the most important upcoming "
        "arrivals and reservations.",

    "vip":
        "Which VIP or special requests require "
        "attention?",

    "staffing":
        "Where are the most important staffing "
        "shortages or coverage gaps?",

    "management":
        "What are the three most important actions "
        "management should take today?"
}


# ============================================================
# PROCESS BUTTON QUESTION
# ============================================================

def process_question(question):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    answer = ask_marriott_bot(
        question
    )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# BUTTON LAYOUT
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "📊 Occupancy Situation",
        use_container_width=True
    ):

        process_question(
            QUESTIONS["occupancy"]
        )

        st.rerun()


    if st.button(
        "⭐ Pending VIP Requests",
        use_container_width=True
    ):

        process_question(
            QUESTIONS["vip"]
        )

        st.rerun()


with col2:

    if st.button(
        "🔧 Maintenance Priorities",
        use_container_width=True
    ):

        process_question(
            QUESTIONS["maintenance"]
        )

        st.rerun()


    if st.button(
        "👥 Staffing Shortages",
        use_container_width=True
    ):

        process_question(
            QUESTIONS["staffing"]
        )

        st.rerun()


with col3:

    if st.button(
        "🧳 Arrivals and Reservations",
        use_container_width=True
    ):

        process_question(
            QUESTIONS["reservations"]
        )

        st.rerun()


    if st.button(
        "🚨 Top 3 Management Actions",
        use_container_width=True
    ):

        process_question(
            QUESTIONS["management"]
        )

        st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask about hotel operations..."
)


if user_question:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # Display user message
    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing hotel operations..."
        ):

            answer = ask_marriott_bot(
                user_question
            )

        st.markdown(
            answer
        )


    # Store answer
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
