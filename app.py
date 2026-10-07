
import streamlit as st
import pandas as pd
from google import genai

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="Marriott Morning Briefing Bot",
    page_icon="🏨",
    layout="wide"
)

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

# -------------------------------------------------
# GEMINI CONNECTION
# -------------------------------------------------

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error(
        "Gemini API key is not configured. "
        "Add GEMINI_API_KEY to Streamlit Secrets."
    )
    st.stop()

# -------------------------------------------------
# LOAD EXCEL DATA
# -------------------------------------------------

FILE_NAME = "Marriott_Morning_Briefing_Bot.xlsx"

@st.cache_data
def load_data():

    occupancy = pd.read_excel(
        FILE_NAME,
        sheet_name="Occupancy",
        header=3
    )

    reservations = pd.read_excel(
        FILE_NAME,
        sheet_name="Reservations",
        header=3
    )

    vip = pd.read_excel(
        FILE_NAME,
        sheet_name="VIP Requests",
        header=3
    )

    maintenance = pd.read_excel(
        FILE_NAME,
        sheet_name="HK Maintenance",
        header=3
    )

    staffing = pd.read_excel(
        FILE_NAME,
        sheet_name="Staffing Roster",
        header=3
    )

    reviews = pd.read_excel(
        FILE_NAME,
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


try:

    (
        occupancy,
        reservations,
        vip,
        maintenance,
        staffing,
        reviews
    ) = load_data()

except Exception as error:

    st.error(
        "Unable to load the Marriott Excel dataset."
    )

    st.exception(error)
    st.stop()

# -------------------------------------------------
# PREPARE DATA
# -------------------------------------------------

def prepare_data(df):

    temp = df.copy()

    for column in temp.columns:
        temp[column] = temp[column].astype(str)

    return temp.to_csv(index=False)


hotel_data = f"""
=== OCCUPANCY ===
{prepare_data(occupancy)}

=== RESERVATIONS ===
{prepare_data(reservations)}

=== VIP REQUESTS ===
{prepare_data(vip)}

=== HOUSEKEEPING AND MAINTENANCE ===
{prepare_data(maintenance)}

=== STAFFING ROSTER ===
{prepare_data(staffing)}

=== GUEST REVIEWS ===
{prepare_data(reviews)}
"""

# -------------------------------------------------
# CHATBOT
# -------------------------------------------------

def ask_marriott_bot(question):

    prompt = f"""
You are the Marriott Morning Briefing Bot.

This application is an academic project using simulated
hotel data. The records are not actual Marriott
International operational information.

Your role is to help a hotel manager interpret daily
hotel operations.

RULES:

1. Use ONLY the supplied hotel dataset.
2. Never invent numbers, guests, requests or incidents.
3. If information is unavailable, clearly say so.
4. Prioritize critical and high-priority maintenance.
5. Identify staffing shortages when relevant.
6. Identify pending VIP and special requests.
7. Analyze occupancy and reservations when relevant.
8. Consider guest reviews and unresolved responses.
9. Provide practical management recommendations.
10. Keep answers concise, professional and operational.
11. Explain why urgent actions should be prioritized.

HOTEL DATA:

{hotel_data}

MANAGER QUESTION:

{question}

MANAGEMENT RESPONSE:
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        "This chatbot analyzes a simulated 30-day "
        "Marriott hotel operations dataset."
    )

    st.write("Data analyzed:")

    st.write("• Occupancy")
    st.write("• Reservations")
    st.write("• VIP requests")
    st.write("• Maintenance")
    st.write("• Staffing")
    st.write("• Guest reviews")

    st.divider()

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# -------------------------------------------------
# SUGGESTED QUESTIONS
# -------------------------------------------------

st.subheader("Suggested Questions")

col1, col2, col3 = st.columns(3)

with col1:

    occupancy_button = st.button(
        "📊 Occupancy Situation"
    )

    vip_button = st.button(
        "⭐ Pending VIP Requests"
    )

with col2:

    maintenance_button = st.button(
        "🔧 Maintenance Priorities"
    )

    staffing_button = st.button(
        "👥 Staffing Shortages"
    )

with col3:

    arrivals_button = st.button(
        "🧳 Arrivals and Reservations"
    )

    actions_button = st.button(
        "🚨 Top 3 Management Actions"
    )

# -------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content":
            "Welcome to the Marriott Morning Briefing Bot. "
            "Ask me about hotel operations."
        }
    ]


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# HANDLE SUGGESTIONS
# -------------------------------------------------

suggested_question = None

if occupancy_button:
    suggested_question = (
        "What is the occupancy situation in the hotel?"
    )

elif vip_button:
    suggested_question = (
        "Which VIP requests are still incomplete?"
    )

elif maintenance_button:
    suggested_question = (
        "Which maintenance issues require immediate attention?"
    )

elif staffing_button:
    suggested_question = (
        "Where are the staffing shortages?"
    )

elif arrivals_button:
    suggested_question = (
        "Summarize the important reservation and arrival activity."
    )

elif actions_button:
    suggested_question = (
        "What are the three most important actions "
        "management should take?"
    )

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------

typed_question = st.chat_input(
    "Ask about hotel operations..."
)

question = typed_question or suggested_question

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing hotel operations..."
        ):

            try:

                answer = ask_marriott_bot(question)

                st.markdown(answer)

            except Exception as error:

                answer = (
                    "I encountered an error while analyzing "
                    "the hotel data."
                )

                st.error(answer)
                st.exception(error)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
