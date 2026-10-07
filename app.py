import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import re
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Marriott Morning Briefing Bot",
    page_icon="🏨",
    layout="wide"
)

# =========================================================
# SETTINGS
# =========================================================

FILE_NAME = "Marriott_Morning_Briefing_Bot.xlsx"

# Gemini is ONLY used for questions that cannot be answered
# directly from the hotel dataset.
GEMINI_MODEL = "gemini-3.8-flash"


# =========================================================
# API KEY
# =========================================================

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


API_KEY = get_api_key()

if API_KEY:
    gemini_client = genai.Client(api_key=API_KEY)
else:
    gemini_client = None


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
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

    # Clean column names
    dataframes = [
        occupancy,
        reservations,
        vip,
        maintenance,
        staffing,
        reviews
    ]

    for df in dataframes:
        df.columns = [str(c).strip() for c in df.columns]
        df.dropna(how="all", inplace=True)

    return {
        "occupancy": occupancy,
        "reservations": reservations,
        "vip": vip,
        "maintenance": maintenance,
        "staffing": staffing,
        "reviews": reviews
    }


try:
    data = load_data()

except Exception as e:

    st.error("Unable to load the Marriott Excel dataset.")

    st.error(str(e))

    st.info(
        "Make sure Marriott_Morning_Briefing_Bot.xlsx "
        "is uploaded to the same GitHub repository as app.py."
    )

    st.stop()


occupancy = data["occupancy"]
reservations = data["reservations"]
vip = data["vip"]
maintenance = data["maintenance"]
staffing = data["staffing"]
reviews = data["reviews"]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_text(value):
    if pd.isna(value):
        return "N/A"
    return str(value)


def find_column(df, possible_names):

    for col in df.columns:

        normalized = str(col).lower().strip()

        for name in possible_names:

            if name.lower() in normalized:
                return col

    return None


def format_date(value):

    if pd.isna(value):
        return "N/A"

    try:
        return pd.to_datetime(value).strftime("%B %d, %Y")
    except Exception:
        return str(value)


# =========================================================
# OCCUPANCY ANALYSIS
# =========================================================

def occupancy_answer():

    occ_col = find_column(
        occupancy,
        ["occupancy %", "occupancy"]
    )

    date_col = find_column(
        occupancy,
        ["date"]
    )

    booked_col = find_column(
        occupancy,
        ["rooms booked"]
    )

    available_col = find_column(
        occupancy,
        ["available rooms"]
    )

    if occ_col is None:
        return "Occupancy information is unavailable."

    df = occupancy.copy()

    df[occ_col] = pd.to_numeric(
        df[occ_col],
        errors="coerce"
    )

    df = df.dropna(subset=[occ_col])

    if df.empty:
        return "No valid occupancy records were found."

    # Handle percentage stored as decimal
    if df[occ_col].max() <= 1.5:
        df["_occupancy_display"] = df[occ_col] * 100
    else:
        df["_occupancy_display"] = df[occ_col]

    avg = df["_occupancy_display"].mean()

    top = df.nlargest(
        min(5, len(df)),
        "_occupancy_display"
    )

    answer = (
        f"### 📊 Occupancy Situation\n\n"
        f"**Average occupancy:** {avg:.1f}%\n\n"
        f"**Highest-demand dates requiring management attention:**\n\n"
    )

    for _, row in top.iterrows():

        date = format_date(row[date_col]) if date_col else "N/A"

        occ = row["_occupancy_display"]

        answer += f"- **{date}: {occ:.1f}% occupancy**"

        if booked_col:
            answer += f" | Rooms booked: {safe_text(row[booked_col])}"

        if available_col:
            answer += f" | Available: {safe_text(row[available_col])}"

        answer += "\n"

    answer += (
        "\n**Management recommendation:** Prioritize staffing, "
        "room readiness, maintenance clearance, and VIP preparation "
        "on the highest-occupancy dates."
    )

    return answer


# =========================================================
# VIP ANALYSIS
# =========================================================

def vip_answer():

    status_col = find_column(vip, ["status"])

    request_col = find_column(
        vip,
        ["request type"]
    )

    guest_col = find_column(
        vip,
        ["guest"]
    )

    owner_col = find_column(
        vip,
        ["owner"]
    )

    id_col = find_column(
        vip,
        ["request id"]
    )

    if status_col is None:
        return "VIP request status information is unavailable."

    active = vip[
        ~vip[status_col]
        .astype(str)
        .str.lower()
        .str.contains(
            "complete|completed|closed|resolved",
            na=False
        )
    ].copy()

    if active.empty:
        return (
            "### ⭐ VIP Requests\n\n"
            "There are currently no outstanding VIP requests."
        )

    answer = (
        f"### ⭐ Pending VIP Requests\n\n"
        f"There are **{len(active)} outstanding VIP or special requests**.\n\n"
    )

    for _, row in active.head(10).iterrows():

        req_id = safe_text(row[id_col]) if id_col else ""
        guest = safe_text(row[guest_col]) if guest_col else "Guest"
        request = safe_text(row[request_col]) if request_col else "Request"
        status = safe_text(row[status_col])
        owner = safe_text(row[owner_col]) if owner_col else "Unassigned"

        answer += (
            f"- **{req_id} – {guest}:** "
            f"{request} | {status} | Owner: {owner}\n"
        )

    answer += (
        "\n**Management recommendation:** Resolve requests associated "
        "with arriving guests first and confirm ownership of every "
        "pending request."
    )

    return answer


# =========================================================
# MAINTENANCE ANALYSIS
# =========================================================

def maintenance_answer():

    status_col = find_column(
        maintenance,
        ["status"]
    )

    priority_col = find_column(
        maintenance,
        ["priority"]
    )

    room_col = find_column(
        maintenance,
        ["room"]
    )

    category_col = find_column(
        maintenance,
        ["category"]
    )

    description_col = find_column(
        maintenance,
        ["description"]
    )

    id_col = find_column(
        maintenance,
        ["issue id"]
    )

    if status_col is None:
        return "Maintenance status information is unavailable."

    open_items = maintenance[
        ~maintenance[status_col]
        .astype(str)
        .str.lower()
        .str.contains(
            "complete|completed|closed|resolved",
            na=False
        )
    ].copy()

    if priority_col:

        priority_order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3
        }

        open_items["_priority"] = (
            open_items[priority_col]
            .astype(str)
            .str.lower()
            .map(priority_order)
            .fillna(4)
        )

        open_items = open_items.sort_values("_priority")

    answer = (
        f"### 🔧 Maintenance Priorities\n\n"
        f"There are **{len(open_items)} open maintenance issues**.\n\n"
    )

    if open_items.empty:

        answer += "No outstanding maintenance issues were found."

        return answer

    for _, row in open_items.head(10).iterrows():

        issue = safe_text(row[id_col]) if id_col else ""
        room = safe_text(row[room_col]) if room_col else "N/A"
        priority = safe_text(row[priority_col]) if priority_col else "N/A"
        category = safe_text(row[category_col]) if category_col else ""
        description = (
            safe_text(row[description_col])
            if description_col else ""
        )

        answer += (
            f"- **{issue} | Room {room} | {priority}:** "
            f"{category}"
        )

        if description and description != "N/A":
            answer += f" – {description}"

        answer += "\n"

    answer += (
        "\n**Management recommendation:** Address Critical and High "
        "priority issues first, especially those affecting occupied "
        "or soon-to-be-occupied rooms."
    )

    return answer


# =========================================================
# STAFFING ANALYSIS
# =========================================================

def staffing_answer():

    gap_col = find_column(
        staffing,
        ["gap"]
    )

    date_col = find_column(
        staffing,
        ["date"]
    )

    shift_col = find_column(
        staffing,
        ["shift"]
    )

    role_col = find_column(
        staffing,
        ["role"]
    )

    assigned_col = find_column(
        staffing,
        ["staff assigned"]
    )

    required_col = find_column(
        staffing,
        ["staff required"]
    )

    if gap_col is None:
        return "Staffing gap information is unavailable."

    df = staffing.copy()

    df[gap_col] = pd.to_numeric(
        df[gap_col],
        errors="coerce"
    ).fillna(0)

    # Some spreadsheets represent shortages as positive;
    # others as negative.
    shortages = df[df[gap_col] != 0].copy()

    answer = (
        f"### 👥 Staffing Shortages\n\n"
        f"**{len(shortages)} staffing records contain a gap.**\n\n"
    )

    if shortages.empty:

        answer += "No staffing shortages were identified."

        return answer

    shortages["_abs_gap"] = shortages[gap_col].abs()

    shortages = shortages.sort_values(
        "_abs_gap",
        ascending=False
    )

    for _, row in shortages.head(10).iterrows():

        date = format_date(row[date_col]) if date_col else "N/A"
        shift = safe_text(row[shift_col]) if shift_col else "N/A"
        role = safe_text(row[role_col]) if role_col else "N/A"

        answer += f"- **{date} | {shift} | {role}**"

        if assigned_col and required_col:

            answer += (
                f" – Assigned: {safe_text(row[assigned_col])}, "
                f"Required: {safe_text(row[required_col])}"
            )

        answer += (
            f" | Gap: {safe_text(row[gap_col])}\n"
        )

    answer += (
        "\n**Management recommendation:** Reallocate available employees "
        "or cross-train staff for the largest operational gaps, "
        "particularly on high-occupancy dates."
    )

    return answer


# =========================================================
# RESERVATION ANALYSIS
# =========================================================

def reservation_answer():

    checkin_col = find_column(
        reservations,
        ["check-in date", "check in date"]
    )

    guest_col = find_column(
        reservations,
        ["guest name"]
    )

    status_col = find_column(
        reservations,
        ["status"]
    )

    room_col = find_column(
        reservations,
        ["room type"]
    )

    guests_col = find_column(
        reservations,
        ["guests"]
    )

    answer = (
        f"### 🧳 Arrivals and Reservations\n\n"
        f"The dataset contains **{len(reservations)} reservations**.\n\n"
    )

    df = reservations.copy()

    if checkin_col:

        df["_checkin"] = pd.to_datetime(
            df[checkin_col],
            errors="coerce"
        )

        df = df.sort_values("_checkin")

    answer += "**Upcoming reservation records:**\n\n"

    for _, row in df.head(10).iterrows():

        guest = (
            safe_text(row[guest_col])
            if guest_col else "Guest"
        )

        date = (
            format_date(row[checkin_col])
            if checkin_col else "N/A"
        )

        answer += f"- **{guest}** – Check-in: {date}"

        if room_col:
            answer += f" | {safe_text(row[room_col])}"

        if guests_col:
            answer += f" | Guests: {safe_text(row[guests_col])}"

        if status_col:
            answer += f" | Status: {safe_text(row[status_col])}"

        answer += "\n"

    answer += (
        "\n**Management recommendation:** Coordinate Front Office and "
        "Housekeeping around concentrated arrival periods and verify "
        "special requests before arrival."
    )

    return answer


# =========================================================
# REVIEWS ANALYSIS
# =========================================================

def reviews_answer():

    rating_col = find_column(
        reviews,
        ["rating"]
    )

    text_col = find_column(
        reviews,
        ["review text"]
    )

    category_col = find_column(
        reviews,
        ["category"]
    )

    response_col = find_column(
        reviews,
        ["response status"]
    )

    answer = "### 💬 Guest Review Summary\n\n"

    if rating_col:

        ratings = pd.to_numeric(
            reviews[rating_col],
            errors="coerce"
        )

        answer += (
            f"**Average guest rating:** "
            f"{ratings.mean():.2f}\n\n"
        )

        negative = reviews[ratings <= 3].copy()

    else:
        negative = reviews.copy()

    answer += (
        f"**Reviews requiring attention:** "
        f"{len(negative)}\n\n"
    )

    for _, row in negative.head(8).iterrows():

        category = (
            safe_text(row[category_col])
            if category_col else "Guest feedback"
        )

        text = (
            safe_text(row[text_col])
            if text_col else ""
        )

        answer += f"- **{category}:** {text}"

        if response_col:

            answer += (
                f" | Response: "
                f"{safe_text(row[response_col])}"
            )

        answer += "\n"

    answer += (
        "\n**Management recommendation:** Prioritize low-rating reviews "
        "and unresolved complaints for immediate service recovery."
    )

    return answer


# =========================================================
# MANAGEMENT PRIORITIES
# =========================================================

def management_answer():

    maintenance_status = find_column(
        maintenance,
        ["status"]
    )

    vip_status = find_column(
        vip,
        ["status"]
    )

    staffing_gap = find_column(
        staffing,
        ["gap"]
    )

    open_maintenance = 0
    open_vip = 0
    staff_gaps = 0

    if maintenance_status:

        open_maintenance = len(
            maintenance[
                ~maintenance[maintenance_status]
                .astype(str)
                .str.lower()
                .str.contains(
                    "complete|closed|resolved",
                    na=False
                )
            ]
        )

    if vip_status:

        open_vip = len(
            vip[
                ~vip[vip_status]
                .astype(str)
                .str.lower()
                .str.contains(
                    "complete|closed|resolved",
                    na=False
                )
            ]
        )

    if staffing_gap:

        gaps = pd.to_numeric(
            staffing[staffing_gap],
            errors="coerce"
        ).fillna(0)

        staff_gaps = int((gaps != 0).sum())

    return f"""
### 🚨 Top 3 Management Actions

**1. Resolve maintenance risks**

There are **{open_maintenance} open maintenance records**. Critical and High priority issues should be reviewed first so room availability, safety, and guest experience are protected.

**2. Close VIP and special-request backlogs**

There are **{open_vip} outstanding VIP or special requests**. Management should verify ownership, prioritize arriving guests, and confirm completion before check-in.

**3. Stabilize staffing coverage**

There are **{staff_gaps} staffing records with identified gaps**. Management should compare these shortages with high-occupancy dates and reallocate or cross-train staff where necessary.

**Overall priority:** Align staffing, maintenance, VIP preparation, and room readiness with the hotel's highest-demand dates.
"""


# =========================================================
# DATA SUMMARY FOR GEMINI
# =========================================================

@st.cache_data(show_spinner=False)
def create_ai_context():

    # Send only small samples and summaries.
    # This makes Gemini much faster than sending the whole workbook.

    context = f"""
SIMULATED MARRIOTT HOTEL OPERATIONS DATA

IMPORTANT:
This is simulated academic data.
It is not real Marriott International operational information.

OCCUPANCY
Rows: {len(occupancy)}
Columns: {list(occupancy.columns)}
Sample:
{occupancy.head(8).to_string(index=False)}

RESERVATIONS
Rows: {len(reservations)}
Columns: {list(reservations.columns)}
Sample:
{reservations.head(8).to_string(index=False)}

VIP REQUESTS
Rows: {len(vip)}
Columns: {list(vip.columns)}
Sample:
{vip.head(10).to_string(index=False)}

MAINTENANCE
Rows: {len(maintenance)}
Columns: {list(maintenance.columns)}
Sample:
{maintenance.head(10).to_string(index=False)}

STAFFING
Rows: {len(staffing)}
Columns: {list(staffing.columns)}
Sample:
{staffing.head(10).to_string(index=False)}

GUEST REVIEWS
Rows: {len(reviews)}
Columns: {list(reviews.columns)}
Sample:
{reviews.head(8).to_string(index=False)}
"""

    return context


AI_CONTEXT = create_ai_context()


# =========================================================
# QUESTION ROUTER
# =========================================================

def route_question(question):

    q = question.lower().strip()

    # Local answers = extremely fast

    if any(
        word in q
        for word in [
            "occupancy",
            "occupied",
            "room demand",
            "highest demand",
            "busiest date"
        ]
    ):
        return occupancy_answer()

    if any(
        word in q
        for word in [
            "vip",
            "special request",
            "pending request"
        ]
    ):
        return vip_answer()

    if any(
        word in q
        for word in [
            "maintenance",
            "repair",
            "issue",
            "housekeeping issue"
        ]
    ):
        return maintenance_answer()

    if any(
        word in q
        for word in [
            "staff",
            "staffing",
            "shortage",
            "coverage",
            "roster"
        ]
    ):
        return staffing_answer()

    if any(
        word in q
        for word in [
            "reservation",
            "arrival",
            "check-in",
            "check in",
            "booking"
        ]
    ):
        return reservation_answer()

    if any(
        word in q
        for word in [
            "review",
            "rating",
            "complaint",
            "guest feedback"
        ]
    ):
        return reviews_answer()

    if any(
        phrase in q
        for phrase in [
            "top 3",
            "top three",
            "management action",
            "management priority",
            "management priorities",
            "most important action",
            "what should management"
        ]
    ):
        return management_answer()

    # Otherwise use Gemini
    return ask_gemini(question)


# =========================================================
# GEMINI FALLBACK
# =========================================================

def ask_gemini(question):

    if gemini_client is None:

        return (
            "Gemini is not configured. The built-in hotel analysis "
            "still works for occupancy, reservations, VIP requests, "
            "maintenance, staffing, guest reviews, and management priorities."
        )

    prompt = f"""
You are the Marriott Morning Briefing Bot for an academic project.

Use ONLY the simulated hotel information supplied below.

Do not claim that this is real Marriott International operational data.
Do not invent records, dates, guests, statistics, or operational facts.

Answer the manager's question directly.

Keep the response concise and operational.

Use:
1. Key finding
2. Important evidence
3. Recommended management action

HOTEL DATA:

{AI_CONTEXT}

MANAGER QUESTION:

{question}
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=450,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            )
        )

        if response and response.text:
            return response.text

        return (
            "The AI returned an empty response. "
            "Please try the question again."
        )

    except Exception:

        return (
            "The Gemini service is temporarily unavailable. "
            "The hotel's built-in analytical functions are still working. "
            "Try one of the suggested questions above."
        )


# =========================================================
# USER INTERFACE
# =========================================================

st.title("🏨 Marriott Morning Briefing Bot")

st.caption(
    "AI-assisted hotel operations briefing | "
    "Academic project using simulated Marriott data"
)

st.warning(
    "All hotel records used in this application are simulated "
    "for academic purposes and are not actual Marriott International "
    "operational data."
)


# =========================================================
# SIDEBAR
# =========================================================

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

    st.success("⚡ Fast local analysis enabled")

    if gemini_client:
        st.success("🤖 Gemini AI connected")
    else:
        st.warning("Gemini API not connected")

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content":
                "Welcome to the Marriott Morning Briefing Bot. "
                "Ask me about hotel operations."
        }
    ]


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

st.subheader("Suggested Questions")

col1, col2, col3 = st.columns(3)

suggested_question = None


with col1:

    if st.button(
        "📊 Occupancy Situation",
        use_container_width=True
    ):
        suggested_question = (
            "What is the occupancy situation and which dates "
            "require the most management attention?"
        )

    if st.button(
        "⭐ Pending VIP Requests",
        use_container_width=True
    ):
        suggested_question = (
            "Which VIP requests are still pending?"
        )


with col2:

    if st.button(
        "🔧 Maintenance Priorities",
        use_container_width=True
    ):
        suggested_question = (
            "What maintenance issues should management prioritize?"
        )

    if st.button(
        "👥 Staffing Shortages",
        use_container_width=True
    ):
        suggested_question = (
            "Where are the most important staffing shortages?"
        )


with col3:

    if st.button(
        "🧳 Arrivals and Reservations",
        use_container_width=True
    ):
        suggested_question = (
            "Summarize the important arrivals and reservations."
        )

    if st.button(
        "🚨 Top 3 Management Actions",
        use_container_width=True
    ):
        suggested_question = (
            "What are the top three management priorities?"
        )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

typed_question = st.chat_input(
    "Ask about hotel operations..."
)

question = suggested_question or typed_question


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Most questions now execute locally,
    # so there is no long spinner.

    with st.chat_message("assistant"):

        answer = route_question(question)

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
