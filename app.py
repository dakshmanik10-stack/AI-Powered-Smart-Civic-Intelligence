from datetime import date, datetime
import sqlite3
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATABASE_PATH = Path(__file__).with_name("civic_complaints.db")

st.set_page_config(
    page_title="Civic Issue Reporting",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORIES = [
    "Abandoned vehicles on roads",
    "Contaminated water",
    "Dangerous hanging wires",
    "Faulty traffic lights",
    "Garbage/Cleaning",
    "Illegal Parking",
    "Illegal construction",
    "Manholes/Sewer blockages",
    "Other",
    "Pipeline leaks",
    "Potholes",
    "Public Parks",
    "Streetlights",
    "Transformer sparks",
    "Water supply disruption",
]

DEMO_USERS = {
    "citizen": {
        "password": "citizen123",
        "name": "Shashi Ranjan",
        "citizen_id": "CIT-1001",
        "role": "Citizen",
    },
    "rahul": {
        "password": "rahul123",
        "name": "Rahul Kumar",
        "citizen_id": "CIT-1002",
        "role": "Citizen",
    },
    "admin": {
        "password": "admin123",
        "name": "Admin",
        "citizen_id": "ADMIN-001",
        "role": "Admin",
    },
}

MOCK_REPORTS = [
    {
        "id": "CVS-100001-2023",
        "date": date(2026, 9, 19),
        "name": "Shashi Ranjan",
        "citizen_id": "CIT-1001",
        "category": "Potholes",
        "description": "A large pothole is making the road unsafe near the main crossing.",
        "status": "Pending",
        "image": None,
        "video": None,
        "location": "Main crossing, Connaught Place",
        "pincode": "110001",
        "upvotes": 1,
        "downvotes": 0,
    },
    {
        "id": "CVS-100002-2023",
        "date": date(2026, 9, 3),
        "name": "Shashi Ranjan",
        "citizen_id": "CIT-1001",
        "category": "Water supply disruption",
        "description": "There has been water leakage near the community park.",
        "status": "Pending",
        "image": None,
        "video": None,
        "location": "Community park, Connaught Place",
        "pincode": "110001",
        "upvotes": 0,
        "downvotes": 0,
    },
    {
        "id": "CVS-100003-2023",
        "date": date(2026, 8, 27),
        "name": "Rahul Kumar",
        "citizen_id": "CIT-1002",
        "category": "Streetlights",
        "description": "Streetlight not working on the road beside the library.",
        "status": "In Progress",
        "image": None,
        "video": None,
        "location": "Library Road, Karol Bagh",
        "pincode": "110005",
        "upvotes": 2,
        "downvotes": 0,
    },
    {
        "id": "CVS-100004-2023",
        "date": date(2026, 8, 14),
        "name": "Rahul Kumar",
        "citizen_id": "CIT-1002",
        "category": "Garbage/Cleaning",
        "description": "Garbage has not been collected from this street for several days.",
        "status": "Resolved",
        "image": None,
        "video": None,
        "location": "Market Street, Karol Bagh",
        "pincode": "110005",
        "upvotes": 1,
        "downvotes": 0,
    },
]


def save_report(report):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO complaints (
                id, date, name, citizen_id, category, description, status,
                image, video, location, pincode, upvotes, downvotes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["id"],
                report["date"].isoformat(),
                report["name"],
                report["citizen_id"],
                report["category"],
                report["description"],
                report["status"],
                report.get("image"),
                report.get("video"),
                report.get("location", ""),
                report.get("pincode", ""),
                report.get("upvotes", 0),
                report.get("downvotes", 0),
            ),
        )


def load_reports():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM complaints ORDER BY date DESC, rowid DESC"
        ).fetchall()

    reports = []
    for row in rows:
        report = dict(row)
        report["date"] = date.fromisoformat(report["date"])
        reports.append(report)
    return reports


def update_report_statuses(status_by_report_id):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.executemany(
            "UPDATE complaints SET status = ? WHERE id = ?",
            [
                (status, report_id)
                for report_id, status in status_by_report_id.items()
            ],
        )


def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                name TEXT NOT NULL,
                citizen_id TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                image BLOB,
                video BLOB,
                location TEXT NOT NULL,
                pincode TEXT NOT NULL,
                upvotes INTEGER NOT NULL DEFAULT 0,
                downvotes INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        complaint_count = connection.execute(
            "SELECT COUNT(*) FROM complaints"
        ).fetchone()[0]
        if complaint_count == 0:
            connection.executemany(
                """
                INSERT INTO complaints (
                    id, date, name, citizen_id, category, description, status,
                    image, video, location, pincode, upvotes, downvotes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        report["id"],
                        report["date"].isoformat(),
                        report["name"],
                        report["citizen_id"],
                        report["category"],
                        report["description"],
                        report["status"],
                        report.get("image"),
                        report.get("video"),
                        report.get("location", ""),
                        report.get("pincode", ""),
                        report.get("upvotes", 0),
                        report.get("downvotes", 0),
                    )
                    for report in MOCK_REPORTS
                ],
            )


def initialize_state():
    initialize_database()
    st.session_state.reports = load_reports()
    if "citizen_accounts" not in st.session_state:
        st.session_state.citizen_accounts = {
            username: account.copy()
            for username, account in DEMO_USERS.items()
            if account["role"] == "Citizen"
        }


def add_report(
    category,
    description,
    location,
    pincode,
    uploaded_file,
    user,
):
    attachment = None
    attachment_type = None
    if uploaded_file is not None:
        attachment = uploaded_file.getvalue()
        attachment_type = uploaded_file.type

    report = {
        "id": f"CVS-{uuid4().hex[:6].upper()}-{datetime.now():%Y}",
        "date": date.today(),
        "name": user["name"],
        "citizen_id": user["citizen_id"],
        "category": category,
        "description": description.strip(),
        "status": "Pending",
        "image": attachment if attachment_type and attachment_type.startswith("image/") else None,
        "video": attachment if attachment_type and attachment_type.startswith("video/") else None,
        "location": location.strip(),
        "pincode": pincode.strip(),
        "upvotes": 0,
        "downvotes": 0,
    }
    save_report(report)
    st.session_state.reports.insert(0, report)


def render_login():
    st.title("Welcome to the Civic Portal")
    st.caption("Sign in to report civic issues or manage complaints.")
    left, login_col, right = st.columns([1, 1.2, 1])
    with login_col:
        login_tab, signup_tab = st.tabs(["Log in", "Citizen Sign up"])
        with login_tab:
            with st.container(border=True):
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button(
                        "Log in",
                        type="primary",
                        use_container_width=True,
                    )
                if submitted:
                    normalized_username = username.strip().lower()
                    if normalized_username == "admin":
                        account = DEMO_USERS["admin"]
                    else:
                        account = st.session_state.citizen_accounts.get(
                            normalized_username
                        )

                    if account and account["password"] == password:
                        st.session_state.current_user = {
                            "username": normalized_username,
                            "name": account["name"],
                            "citizen_id": account["citizen_id"],
                            "role": account["role"],
                        }
                        st.rerun()
                    st.error("Username or password is incorrect.")
                st.caption(
                    "Admin login: username `admin` and the admin password. "
                    "All other valid accounts are citizens."
                )

        with signup_tab:
            with st.container(border=True):
                with st.form("citizen_signup_form", clear_on_submit=True):
                    name = st.text_input("Full Name")
                    new_username = st.text_input("Choose a Username")
                    new_password = st.text_input("Create Password", type="password")
                    confirm_password = st.text_input(
                        "Confirm Password",
                        type="password",
                    )
                    signup_submitted = st.form_submit_button(
                        "Create Citizen Account",
                        type="primary",
                        use_container_width=True,
                    )

                if signup_submitted:
                    normalized_name = " ".join(name.split())
                    normalized_username = new_username.strip().lower()
                    valid_name = bool(
                        normalized_name
                        and all(part.isalpha() for part in normalized_name.split())
                    )
                    valid_password = (
                        len(new_password) >= 4
                        and any(character.isalpha() for character in new_password)
                        and any(character.isdigit() for character in new_password)
                    )

                    if not valid_name:
                        st.error("Name may contain letters and spaces only.")
                    elif not normalized_username:
                        st.error("Please choose a username.")
                    elif normalized_username == "admin":
                        st.error("The username 'admin' is reserved.")
                    elif normalized_username in st.session_state.citizen_accounts:
                        st.error("That username is already registered.")
                    elif not valid_password:
                        st.error(
                            "Password must be at least 4 characters and include "
                            "at least one letter and one number."
                        )
                    elif new_password != confirm_password:
                        st.error("The passwords do not match.")
                    else:
                        citizen_number = 1001 + len(st.session_state.citizen_accounts)
                        citizen_id = f"CIT-{citizen_number}"
                        st.session_state.citizen_accounts[normalized_username] = {
                            "password": new_password,
                            "name": normalized_name,
                            "citizen_id": citizen_id,
                            "role": "Citizen",
                        }
                        st.success(
                            f"Citizen account created. Your Citizen ID is {citizen_id}. "
                            "Log in using your username and password."
                        )

        st.caption(
            "Demo accounts are stored in this browser session only. "
            "Use secure authentication and persistent storage before public deployment."
        )


def render_file_complaint():
    st.title("Report issues, hold authorities accountable")
    st.markdown(
        "<p class='hero-copy'>Snap a photo, share your location, and we'll route your "
        "complaint to the right authority.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("File a Complaint")
    st.caption("Describe the issue and we'll route it to the right authority.")
    user = st.session_state.current_user
    st.info(f"Signed in as {user['name']} · Citizen ID: {user['citizen_id']}")
    with st.form("complaint_form"):
        category = st.pills(
            "Category *",
            CATEGORIES,
            selection_mode="single",
            default=CATEGORIES[0],
            key="complaint_category",
        )
        description = st.text_area(
            "Description *",
            placeholder="Describe the issue in detail — what you see, how long it has been there, and any hazard it poses...",
            height=130,
        )
        attachment = st.file_uploader(
            "Photo / Video (optional)",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi"],
            help="Images and videos only. Maximum file size depends on your Streamlit configuration.",
        )
        st.markdown("#### Location")
        location_col, pincode_col = st.columns([3, 1])
        location = location_col.text_input(
            "Location / Address *",
            placeholder="Street, area, landmark, city",
        )
        pincode = pincode_col.text_input(
            "Pincode *",
            placeholder="110001",
            max_chars=6,
        )
        st.info(
            "Government Routing Notice: This civic reporting demo is independent. "
            "Complaints are shown in this app and are not sent to a government agency."
        )
        submitted = st.form_submit_button(
            "Submit Complaint",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not category:
            st.error("Please choose a complaint category.")
        elif not description.strip():
            st.error("Please describe the issue before submitting.")
        elif not location.strip():
            st.error("Please enter the complaint location.")
        elif not pincode.isdigit() or len(pincode) != 6:
            st.error("Please enter a valid 6-digit pincode.")
        else:
            add_report(
                category,
                description,
                location,
                pincode,
                attachment,
                user,
            )
            st.success(
                f"Complaint submitted. Your tracking ID is "
                f"{st.session_state.reports[0]['id']}."
            )


def render_citizen_dashboard():
    user = st.session_state.current_user
    st.title("My Complaints & Status")
    st.caption(
        f"Track complaints filed by you. Your Citizen ID is {user['citizen_id']}."
    )

    reports = [
        report
        for report in st.session_state.reports
        if report["citizen_id"] == user["citizen_id"]
    ]
    counts = {
        status: sum(report["status"] == status for report in reports)
        for status in ("Pending", "In Progress", "Resolved")
    }
    metrics = st.columns(4)
    metrics[0].metric("Pending", counts["Pending"])
    metrics[1].metric("In Progress", counts["In Progress"])
    metrics[2].metric("Resolved", counts["Resolved"])
    metrics[3].metric("Credits", sum(report["upvotes"] for report in reports))

    st.markdown("### Recent Updates")
    if not reports:
        st.info("You have not filed any complaints yet.")
        return

    for report in reports:
        with st.container(border=True):
            top_col, date_col = st.columns([5, 1])
            status = report["status"]
            badge_color = {
                "Pending": "#f97316",
                "In Progress": "#2563eb",
                "Resolved": "#16a34a",
                "Rejected": "#dc2626",
            }.get(status, "#64748b")
            top_col.markdown(
                f"<span class='status-badge' style='color:{badge_color};"
                f"border-color:{badge_color}55;background:{badge_color}12'>"
                f"{status.upper()}</span>",
                unsafe_allow_html=True,
            )
            date_col.caption(report["date"].strftime("%d %b %Y"))
            st.markdown(f"**{report['category']}**")
            st.write(report["description"])
            if report.get("image"):
                st.image(report["image"], use_container_width=True)
            if report.get("video"):
                st.video(report["video"])
            st.caption(
                f"Location: {report.get('location') or '—'} · "
                f"Pincode: {report.get('pincode') or '—'}"
            )
            vote_up, vote_down, tracking = st.columns([1, 1, 5])
            vote_up.button(
                f"⬆️ {report['upvotes']}",
                key=f"up-{report['id']}",
                help="Upvote this complaint (demo placeholder)",
            )
            vote_down.button(
                f"⬇️ {report['downvotes']}",
                key=f"down-{report['id']}",
                help="Downvote this complaint (demo placeholder)",
            )
            tracking.caption(f"Tracking ID: {report['id']}")


def render_recent_complaints():
    reports = st.session_state.reports
    st.title("Recent Complaints")
    st.caption(
        "Review all citizens' complaints and update their status."
    )
    complaint_editor = pd.DataFrame(
        [
            {
                "Report ID": report["id"],
                "Citizen ID": report["citizen_id"],
                "Date": report["date"].strftime("%d %b %Y"),
                "Name": report["name"],
                "Category": report["category"],
                "Complaint": report["description"],
                "Location": report.get("location", ""),
                "Pincode": report.get("pincode", ""),
                "Update Status": report["status"],
            }
            for report in sorted(
                reports,
                key=lambda item: item["date"],
                reverse=True,
            )
        ]
    )
    edited_complaints = st.data_editor(
        complaint_editor,
        hide_index=True,
        use_container_width=True,
        height=520,
        key="admin_complaint_editor",
        column_config={
            "Report ID": st.column_config.TextColumn(disabled=True),
            "Citizen ID": st.column_config.TextColumn(disabled=True),
            "Update Status": st.column_config.SelectboxColumn(
                "Update Status",
                options=["Pending", "In Progress", "Resolved", "Rejected"],
                required=True,
            ),
            "Complaint": st.column_config.TextColumn(disabled=True, width="large"),
            "Location": st.column_config.TextColumn(disabled=True, width="medium"),
        },
        disabled=[
            "Report ID",
            "Citizen ID",
            "Date",
            "Name",
            "Category",
            "Complaint",
            "Location",
            "Pincode",
        ],
    )
    if st.button("Save Status Updates", type="primary", key="save_status_updates"):
        status_by_report_id = edited_complaints.set_index("Report ID")[
            "Update Status"
        ].to_dict()
        update_report_statuses(status_by_report_id)
        st.session_state.status_update_saved = True
        st.session_state.reports = load_reports()
        del st.session_state["admin_complaint_editor"]
        st.rerun()


def render_admin_dashboard():
    reports = st.session_state.reports
    st.title("Overview")
    st.caption(
        "View all citizen complaints, their contact IDs, locations, and current status."
    )

    overview_cols = st.columns([1, 1, 1, 1.5, 1.15])
    pending_count = sum(report["status"] == "Pending" for report in reports)
    progress_count = sum(report["status"] == "In Progress" for report in reports)
    resolved_count = sum(report["status"] == "Resolved" for report in reports)
    overview_cols[0].metric("Pending", pending_count)
    overview_cols[1].metric("In Progress", progress_count)
    overview_cols[2].metric("Resolved", resolved_count)

    with overview_cols[3]:
        service_level = resolved_count / len(reports) * 100 if reports else 0
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=service_level,
                number={"suffix": "%", "font": {"size": 28}},
                title={"text": "Service Level"},
                gauge={
                    "axis": {"range": [0, 100], "visible": False},
                    "bar": {"color": "#16b981", "thickness": 0.25},
                    "bgcolor": "#e8eef5",
                    "borderwidth": 0,
                    "steps": [{"range": [0, 100], "color": "#e8eef5"}],
                },
            )
        )
        gauge.update_layout(
            height=155,
            margin=dict(l=5, r=5, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.caption("Resolved complaints")

    with overview_cols[4]:
        st.markdown("<div class='rating-card'>", unsafe_allow_html=True)
        st.caption("FEEDBACK RATING")
        st.markdown("## ⭐ 4.6 <small>/ 5</small>", unsafe_allow_html=True)
        st.caption("All time")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Total Complaints")
        days = pd.date_range("2023-02-01", periods=14, freq="D")
        trend_df = pd.DataFrame(
            {
                "Date": list(days) * 2,
                "Complaints": [
                    140, 240, 210, 270, 380, 350, 400, 330, 340, 180, 260, 340, 420, 500,
                    100, 180, 160, 180, 340, 300, 320, 250, 260, 140, 220, 290, 360, 430,
                ],
                "Status": ["Pending"] * 14 + ["In Progress"] * 14,
            }
        )
        trend_fig = px.line(
            trend_df,
            x="Date",
            y="Complaints",
            color="Status",
            markers=True,
            color_discrete_map={"Pending": "#1677ff", "In Progress": "#25c6da"},
        )
        trend_fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=15, b=0),
            legend_title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        trend_fig.update_xaxes(showgrid=False, title=None, tickformat="%b %d")
        trend_fig.update_yaxes(gridcolor="#edf0f5", title=None)
        st.plotly_chart(trend_fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Complaints in Category")
        category_counts = pd.Series(
            [report["category"] for report in reports],
            dtype="string",
        ).value_counts().head(6)
        denominator = len(reports) or 1
        category_data = pd.DataFrame(
            {
                "Category": category_counts.index,
                "Percentage": category_counts.to_numpy() / denominator * 100,
            }
        ).sort_values("Percentage")
        category_fig = px.bar(
            category_data,
            x="Percentage",
            y="Category",
            orientation="h",
            text="Percentage",
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        category_fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
            cliponaxis=False,
        )
        category_fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=5, r=35, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        category_fig.update_xaxes(range=[0, 100], visible=False, title=None)
        category_fig.update_yaxes(showgrid=False, title=None)
        st.plotly_chart(
            category_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )


initialize_state()

st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; color: #172033; }
    [data-testid="stSidebar"] { background: #0847a8; }
    [data-testid="stSidebar"] * { color: #fff; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 9px; padding: .45rem .55rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.14);
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: #0b397f;
        border: 1px solid rgba(255,255,255,.65);
        color: #fff;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: #fff;
        border-color: #fff;
        color: #0847a8;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button p {
        color: inherit;
    }
    h1 { color: #123c32; letter-spacing: -.035em; }
    .hero-copy { color: #52627a; font-size: 1.12rem; margin: -.4rem 0 1.8rem; }
    .status-badge {
        display: inline-block; border: 1px solid; border-radius: 999px;
        padding: .2rem .7rem; font-size: .72rem; font-weight: 700;
        letter-spacing: .04em;
    }
    .rating-card { padding-top: .4rem; text-align: center; }
    .rating-card small { color: #7b8798; font-size: .8rem; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e6ebf2;
        border-radius: 14px; padding: 1rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e4eaf1; border-radius: 14px; background: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "current_user" not in st.session_state:
    render_login()
else:
    current_user = st.session_state.current_user
    st.sidebar.markdown("## 🏙️ Civic Portal")
    if current_user["role"] == "Admin":
        st.sidebar.caption("Admin")
    else:
        st.sidebar.caption(f"Citizen · {current_user['name']}")
        st.sidebar.caption(f"Citizen ID: {current_user['citizen_id']}")
    st.sidebar.markdown("---")

    if current_user["role"] == "Citizen":
        page = st.sidebar.radio(
            "NAVIGATION",
            ["File Complaint", "Track My Complaints"],
            label_visibility="collapsed",
        )
    else:
        page = st.sidebar.radio(
            "NAVIGATION",
            ["Recent Complaints", "Admin GIS Dashboard"],
            label_visibility="collapsed",
        )

    if st.session_state.pop("status_update_saved", False):
        st.success("Complaint statuses updated and saved.")

    if st.sidebar.button("Log out", use_container_width=True):
        del st.session_state["current_user"]
        st.rerun()
    st.sidebar.markdown("---")

    if page == "File Complaint":
        render_file_complaint()
    elif page == "Track My Complaints":
        render_citizen_dashboard()
    elif page == "Recent Complaints":
        render_recent_complaints()
    else:
        render_admin_dashboard()
