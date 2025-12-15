# === Imports ===
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
import re
from shutil import copyfile
import textwrap
from email.message import EmailMessage
from openai import OpenAI
import requests
import base64
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import json
import numpy as np
import time
from collections import defaultdict

# === Utility Functions ===
def wrap_text(text, width=20):
    return '\n'.join(textwrap.wrap(str(text), width=width))

def extract_metadata_from_question(text):
    metadata_pattern = r"\(\s*Distributor:\s*(.*?)\s*;\s*Country:\s*(.*?)\s*;\s*Customer:\s*(.*?)\s*\)\s*$"
    match = re.search(metadata_pattern, text)
    if match:
        distributors = [x.strip() for x in re.split(r",\s*", match.group(1))]
        countries = [x.strip() for x in re.split(r",\s*", match.group(2))]
        customers = [x.strip() for x in re.split(r",\s*", match.group(3))]
        return re.sub(metadata_pattern, "", text).strip(), distributors, countries, customers
    metadata_pattern2 = r"\(\s*Distributor:\s*(.*?)\s*;\s*Country:\s*(.*?)\s*\)\s*$"
    match2 = re.search(metadata_pattern2, text)
    if match2:
        distributors = [x.strip() for x in re.split(r",\s*", match2.group(1))]
        countries = [x.strip() for x in re.split(r",\s*", match2.group(2))]
        customers_set = set()
        for d in distributors:
            for c in countries:
                customers_set.update(distributor_country_to_customers.get((d, c), []))
        return re.sub(metadata_pattern2, "", text).strip(), distributors, countries, sorted(customers_set)
    return re.sub(r"\(.*?\)\s*$", "", text).strip(), ["N/A"], ["N/A"], ["N/A"]

def build_feedback_context(feedback_df, week_num, year):
    """
    TPM-safe: only include Feedback-loop rows from the last 16 ISO weeks.
    Keeps semantics (what was asked/answered, status, entities), but avoids
    huge multi-line blocks.
    """
    # Match your report's anchor (same logic you use elsewhere)
    anchor = datetime.today() + timedelta(days=11)
    allowed_yw = last_n_iso_yearweeks(16, anchor_date=anchor)

    context_lines = []
    for _, row in feedback_df.iterrows():
        row_week = int(pd.to_numeric(row.get("Week"), errors="coerce") or 0)
        row_year = int(pd.to_numeric(row.get("Year"), errors="coerce") or 0)

        # Keep ONLY last 16 ISO weeks
        if (row_year, row_week) not in allowed_yw:
            continue

        am = _clean_one_line(row.get("AM"))
        status = _clean_one_line(row.get("Status"))
        dist = _clean_one_line(row.get("Distributor"))
        country = _clean_one_line(row.get("Country/Countries"))
        cust = _clean_one_line(row.get("Customers"))
        q = _clean_one_line(row.get("Question"))
        ans = _clean_one_line(row.get("Comments / Feedback"))

        # Single-line dense format (much smaller token footprint)
        context_lines.append(
            f"Week {row_week}/{row_year} | AM={am} | Status={status} | "
            f"Distributor={dist} | Country={country} | Customers={cust} | "
            f"Q={q} | A={ans}"
        )

    return "\n".join(context_lines)
def load_last_n_weeks_history(insight_history_path: str, n_weeks: int = 16) -> str:
    """
    Returns only the last N weekly insight blocks from insight_history_path.
    Relies on the delimiter:
      ===== Week {week_num}, {year} =====
    """
    if not os.path.exists(insight_history_path):
        return ""

    with open(insight_history_path, "r", encoding="utf-8") as f:
        txt = f.read()

    parts = re.split(r"(===== Week\s+\d+,\s+\d+\s+=====)", txt)

    blocks = []
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i + 1] if (i + 1) < len(parts) else ""
        blocks.append(header + body)

    return "\n".join(blocks[-n_weeks:]).strip()

def add_comparison_tables_page_to_pdf(pdf, df):
    def draw_comparison_table(ax, data, title, period1_name, period2_name):
        ax.axis('off'); ax.set_title(title, fontsize=12, weight='bold', pad=15)
        if data.empty:
            ax.text(0.5, 0.5, 'No data for this comparison', ha='center', va='center'); return
        data['% Change'] = ((data[period1_name] - data[period2_name]) / data[period2_name] * 100).replace([np.inf, -np.inf], 100).fillna(0)
        display_data = data.copy()
        display_data[period1_name] = display_data[period1_name].apply(lambda x: f'{x:,.0f}')
        display_data[period2_name] = display_data[period2_name].apply(lambda x: f'{x:,.0f}')
        display_data['% Change'] = display_data['% Change'].apply(lambda x: f'{x:+.1f}%')
        display_data = display_data.reset_index()
        table = ax.table(cellText=display_data.values, colLabels=['Product', period1_name, period2_name, '% Change'], loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.04
            if row == 0: cell.set_facecolor("#e6e6fa"); cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
                if col == 3:
                    numeric_value = data.iloc[row - 1]['% Change']
                    color = 'green' if numeric_value > 0.1 else ('red' if numeric_value < -0.1 else 'black')
                    cell.set_text_props(color=color, weight='bold')
                if col == 0: cell.set_text_props(ha='left')
    today = datetime.today() + timedelta(days=11)
    last_week_date = today - timedelta(weeks=1)
    week_before_last_date = today - timedelta(weeks=2)
    same_week_ly_date = last_week_date - timedelta(weeks=52)
    previous_8_weeks = [(d.isocalendar().year, d.isocalendar().week) for d in [today - timedelta(weeks=i) for i in range(1, 9)]]
    before_previous_8_weeks = [(d.isocalendar().year, d.isocalendar().week) for d in [today - timedelta(weeks=i) for i in range(9, 17)]]
    df_agg = lambda yw: df[df['YearWeek'].isin(yw)].groupby('Product')['Total_mCi'].sum() if isinstance(yw, list) else df[df['YearWeek'] == yw].groupby('Product')['Total_mCi'].sum()
    last_week_totals = df_agg((last_week_date.isocalendar().year, last_week_date.isocalendar().week))
    week_before_last_totals = df_agg((week_before_last_date.isocalendar().year, week_before_last_date.isocalendar().week))
    same_week_ly_totals = df_agg((same_week_ly_date.isocalendar().year, same_week_ly_date.isocalendar().week))
    previous_8_weeks_totals = df_agg(previous_8_weeks)
    before_previous_8_weeks_totals = df_agg(before_previous_8_weeks)
    comp1 = pd.concat([last_week_totals.rename('Prev Wk'), week_before_last_totals.rename('Wk Before')], axis=1).fillna(0)
    comp2 = pd.concat([last_week_totals.rename('Prev Wk'), same_week_ly_totals.rename('Same Wk LY')], axis=1).fillna(0)
    comp3 = pd.concat([previous_8_weeks_totals.rename('Prev 8 Wks'), before_previous_8_weeks_totals.rename('8 Wks Before')], axis=1).fillna(0)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5, 11))
    fig.suptitle('Product Performance Summary', fontsize=16, weight='bold')
    draw_comparison_table(ax1, comp1, 'Previous Week vs. Week Before', 'Prev Wk', 'Wk Before')
    draw_comparison_table(ax2, comp2, 'Previous Week vs. Same Week Last Year', 'Prev Wk', 'Same Wk LY')
    draw_comparison_table(ax3, comp3, 'Previous 8 Weeks vs. 8 Weeks Before', 'Prev 8 Wks', '8 Wks Before')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]); pdf.savefig(fig); plt.close(fig)

# --- Setup and Authentication ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1uJAArBkHXgFvY_JPIieKRjiOYd9-Ys7Ii9nXT3fWbUg"
SERVICE_ACCOUNT_FILE = os.path.join(script_dir, 'credentials.json')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)
feedback_ws = gc.open_by_url(SPREADSHEET_URL).worksheet("Feedback loop")
feedback_df = pd.DataFrame(feedback_ws.get_all_records())
feedback_df.columns = [c.strip() for c in feedback_df.columns]
plt.rcParams["font.family"] = "DejaVu Sans"
output_folder = os.path.join(os.getcwd(), 'Reports')
os.makedirs(output_folder, exist_ok=True)
folder_id = "1i1DAOTnF8SznikYrS-ovrg2TRgth9wwP"
SPREADSHEET_URL_DATA = "https://docs.google.com/spreadsheets/d/1WNW4IqAG6W_6XVSaSk9kkukztY93qCDQ3mypZaF5F-Y"
sheet = gc.open_by_url(SPREADSHEET_URL_DATA).worksheet("Airtable Data")
df = pd.DataFrame(sheet.get_all_records())
df.columns = [c.strip() for c in df.columns]
account_manager_map = {"vbeillis@isotopia-global.com": "Vicki Beillis", "naricha@isotopia-global.com": "Noam Aricha", "ndellus@isotopia-global.com": "Noam Dellus", "gbader@isotopia-global.com": "Gilli Bader", "yfarchi@isotopia-global.com": "Yosi Farchi", "caksoy@isotopia-global.com": "Can Aksoy"}
df["Account Manager"] = df["Account Manager Email"].map(account_manager_map).fillna("Other")
def last_n_iso_yearweeks(n_weeks: int, anchor_date: datetime) -> set[tuple[int, int]]:
    """
    Returns a set of (ISO_year, ISO_week) for the last n_weeks,
    anchored to anchor_date.
    """
    yws = set()
    for i in range(n_weeks):
        d = anchor_date - timedelta(weeks=i)
        iso = d.isocalendar()
        yws.add((iso.year, iso.week))
    return yws

def _clean_one_line(s: str) -> str:
    """Collapses whitespace so history lines are token-efficient."""
    if s is None:
        return ""
    return " ".join(str(s).split())

# --- Helper Functions ---
def send_email(subject, body, to_emails, attachment_path):
    if not os.path.exists(attachment_path): print(f"❌ Attachment file not found: {attachment_path}"); return
    with open(attachment_path, "rb") as f:
        files = {'attachments': (os.path.basename(attachment_path), f, 'application/pdf')}
        data = {"from": "Dan Amit <dan.test@resend.dev>", "to": ','.join(to_emails), "subject": subject, "text": body}
        headers = {"Authorization": f"Bearer {os.getenv('RESEND_API_KEY')}"}
        response = requests.post("https://api.resend.com/emails", data=data, files=files, headers=headers)
    if 200 <= response.status_code < 300: print(f"📨 Email sent to: {', '.join(to_emails)}")
    else: print("❌ Failed to send email:", response.status_code, response.text)

def upload_to_drive(file_path, file_name, folder_id=None):
    drive_service = build('drive', 'v3', credentials=creds)
    query = f"name='{file_name}' and trashed=false"
    if folder_id: query += f" and '{folder_id}' in parents"
    existing_files = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute().get('files', [])
    for file in existing_files:
        drive_service.files().delete(fileId=file['id']).execute()
        print(f"🗑️ Deleted: {file['name']} (ID: {file['id']})")
    mimetype = 'application/json' if file_name.endswith('.json') else ('text/plain' if file_name.endswith('.txt') else 'application/pdf')
    media = MediaFileUpload(file_path, mimetype=mimetype)
    file_metadata = {'name': file_name, 'parents': [folder_id] if folder_id else []}
    uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"✅ Uploaded: {file_name} (ID: {uploaded_file.get('id')})")

# --- Data Processing and Analysis ---
today = datetime.today() + timedelta(days=11)
week_num, year = today.isocalendar().week, today.isocalendar().year
df = df.rename(columns={"The customer": "Customer", "Total amount ordered (mCi)": "Total_mCi", "Year": "Year", "Week of supply": "Week", "Shipping Status": "ShippingStatus", "Catalogue description (sold as)": "Product", "Distributing company (from Company name)": "Distributor", "Country": "Country"})
valid_statuses = ["Shipped", "Partially shipped", "Shipped and arrived late", "Order being processed"]
df = df[df["ShippingStatus"].isin(valid_statuses)]
df = df.dropna(subset=["Customer", "Total_mCi", "Year", "Week", "ShippingStatus", "Product"])
for col in ["Total_mCi", "Year", "Week"]: df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Total_mCi", "Year", "Week"])
df["Year"] = df["Year"].astype(int); df["Week"] = df["Week"].astype(int)
df["YearWeek"] = list(zip(df["Year"], df["Week"]))
distributor_country_to_customers = defaultdict(list); country_to_distributors = defaultdict(set); country_to_customers = defaultdict(set); distributor_to_countries = defaultdict(set); distributor_to_customers = defaultdict(set)
for _, row in df.iterrows():
    distributor = str(row['Distributor']).strip(); country = str(row['Country']).strip(); customer = str(row['Customer']).strip()
    if distributor and country and customer:
        distributor_country_to_customers[(distributor, country)].append(customer)
        country_to_distributors[country].add(distributor); country_to_customers[country].add(customer)
        distributor_to_countries[distributor].add(country); distributor_to_customers[distributor].add(customer)
# --- FIXED: Executive Summary Week Calculation (Aligned with PDF) ---

today = datetime.today() + timedelta(days=11)
last_week_date = today - timedelta(weeks=1)

# Build the last 8 COMPLETED ISO weeks (ending with Week 47)
recent_8 = [
    (d.isocalendar().year, d.isocalendar().week)
    for d in [last_week_date - timedelta(weeks=i) for i in range(0, 8)]
][::-1]  # reverse to get oldest→newest

# Build the 8 weeks before those (Weeks 32–39)
previous_8 = [
    (d.isocalendar().year, d.isocalendar().week)
    for d in [last_week_date - timedelta(weeks=i) for i in range(8, 16)]
][::-1]

# Apply these windows
recent_df = df[df["YearWeek"].isin(recent_8)]
previous_df = df[df["YearWeek"].isin(previous_8)]
recent_totals = recent_df.groupby("Customer")["Total_mCi"].sum()
previous_totals = previous_df.groupby("Customer")["Total_mCi"].sum()
customer_to_manager = df.set_index("Customer")["Account Manager"].to_dict()
all_customers = set(df["Customer"])
increased, decreased, stopped = [], [], []
for customer in all_customers:
    prev = previous_totals.get(customer, 0); curr = recent_totals.get(customer, 0); manager = customer_to_manager.get(customer, "Other")
    if prev == 0 and curr > 0: increased.append((customer, prev, curr, manager))
    elif curr == 0 and prev > 0: stopped.append((customer, manager))
    elif curr > prev: increased.append((customer, prev, curr, manager))
    elif curr < prev: decreased.append((customer, prev, curr, manager))
increased_by_am = defaultdict(list); decreased_by_am = defaultdict(list)
for row in increased: increased_by_am[row[3]].append(row)
for row in decreased: decreased_by_am[row[3]].append(row)
customer_to_distributor = df.set_index("Customer")["Distributor"].to_dict()
customer_to_country = df.set_index("Customer")["Country"].to_dict()

# --- ChatGPT Insights & Executive Summary Generation ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=openai_api_key)
summary_path = os.path.join(output_folder, "summary_test.txt")

format_row = lambda name, prev, curr, mgr: (
    f"{name:<30} | {curr - prev:+7.0f} mCi | {(curr - prev) / prev * 100 if prev else 100:+6.1f}% | {mgr}",
    (curr - prev) / prev * 100 if prev else 100
)

increased_formatted = sorted(
    [format_row(*x) for x in increased],
    key=lambda x: x[1],
    reverse=True
)
decreased_formatted = sorted(
    [format_row(*x) for x in decreased],
    key=lambda x: x[1]
)

# === FIX inactive customers logic to match aligned week windows ===
# recent_8 already aligned with PDF (oldest → newest)
# previous_8 same alignment

# Split recent_8 into previous 4 and recent 4 of the aligned window
previous_4_weeks = recent_8[:4]      # weeks 40–43
recent_4_weeks   = recent_8[4:]      # weeks 44–47

active_previous_4 = set(df[df["YearWeek"].isin(previous_4_weeks)]["Customer"])
active_recent_4   = set(df[df["YearWeek"].isin(recent_4_weeks)]["Customer"])

inactive_recent_4 = sorted(active_previous_4 - active_recent_4)

# Build text summary for ChatGPT executive summary input
summary_lines = (
    ["STOPPED ORDERING:", "Customers who stopped ordering in the last 8 weeks...", "---"]
    + [f"{x[0]:<35} | {x[1]}" for x in stopped]
) if stopped else ["- None"]

summary_lines += (
    ["", "DECREASED ORDERS:", "These customers ordered less in the last 8 weeks...", "---"]
    + [x[0] for x in decreased_formatted]
) if decreased_formatted else ["- None"]

summary_lines += (
    ["", "INCREASED ORDERS:", "These customers increased their order amounts...", "---"]
    + [x[0] for x in increased_formatted]
) if increased_formatted else ["- None"]

summary_lines += (
    ["", "INACTIVE IN PAST 4 WEEKS:", "Customers who ordered in the previous 4 weeks...", "---"]
    + inactive_recent_4
) if inactive_recent_4 else ["- None"]

# Save summary file
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

with open(summary_path, "r", encoding="utf-8") as f:
    raw_report_text_for_exec_summary = f.read()

# Include historical insights if exist (DO NOT inject into GPT prompt; archive only)
report_text_with_history = raw_report_text_for_exec_summary
insight_history_path = os.path.join(output_folder, "insight_history_test.txt")

feedback_context = build_feedback_context(feedback_df, week_num, year)

# Prepare distributor metadata for GPT
def format_row_for_gpt(name, prev, curr, mgr):
    distributor = customer_to_distributor.get(name, "Unknown")
    country = customer_to_country.get(name, "Unknown")
    return (
        f"{name:<30} | {curr - prev:+7.0f} mCi | "
        f"{(curr - prev) / prev * 100 if prev else 100:+6.1f}% | "
        f"{mgr} | {distributor} | {country}"
    )

gpt_rows = [
    format_row_for_gpt(name, prev, curr, mgr)
    for (name, prev, curr, mgr) in increased + decreased
]

gpt_distributor_section = (
    ["Customer name | Change | % Change | AM | Distributor | Country"]
    + gpt_rows
)

# Full text sent to GPT
full_prompt_text = (
    "Previous feedback from Account Managers (from the Feedback loop sheet):\n"
    + feedback_context
    + "\n\n==== New Weekly Report Data ====\n\n"
    + report_text_with_history
    + "\n\n=== Distributor info for GPT analysis ===\n"
    + "\n".join(gpt_distributor_section)
)

# === CORRECTED SYSTEM PROMPT AND QUESTION HANDLING LOGIC ===
system_prompt = """
You are a senior business analyst. For each Account Manager (AM), analyze the weekly report using a top-down structure: distributor-level trends, then country-level patterns, then customer-specific insights.

OUTPUT FORMAT (strict):
For each AM, output in this exact order:
1. The AM’s full name on its own line.
2. Either:
   - "No significant insights or questions for this week."
   OR
   - A concise summary grouped exactly as:
     - Distributor-level: key patterns, risks, deviations, and expected order behavior.
     - Country-level: trends affecting multiple customers in the same country.
     - Customer-specific: notable increases, decreases, spikes, drops, or inactivity.
3. A section titled exactly:
   "Questions for <AM Name>:"
   - Include 0–2 numbered questions.
   - Each question must be on one line and end with a metadata block exactly in this format:
     (Distributor: ...; Country: ...; Customer: ...)

REDUNDANCY AND NOVELTY RULES:
Do not repeat questions that were already asked and fully answered in recent weeks unless a new deviation or anomaly exists.
If a customer's ordering behavior remains consistent with a previously confirmed explanation, state that no follow-up is required.
A question is allowed only if it introduces at least one NEW element:
- new customer behavior pattern
- new competitor
- new regulatory status
- new logistics constraint
- new pricing dynamic
- new forecast deviation
If no new element exists, the question must be suppressed.

RESOLUTION STATE CLASSIFICATION (required before questioning):
For recurring customers, classify the current situation as one of the following:
- OPEN: root cause unknown or contradictory.
- EXPLAINED–TEMPORARY: cause known and expected to self-correct within 2–6 weeks.
- EXPLAINED–STRUCTURAL: cause confirmed as cyclical, contractual, batching-based, or trial-scheduled.
- CONTROLLED: operational fix or mitigation has been implemented.
- STRATEGIC RISK: impact driven by competitor displacement or regulatory exclusion.

STATE-BASED QUESTION RULES:
If a customer is EXPLAINED–STRUCTURAL or CONTROLLED, do not generate a question unless:
(a) the deviation exceeds ±30% outside the known pattern, or
(b) a new external factor appears (regulatory, war, competitor exclusivity).
If a customer is EXPLAINED–TEMPORARY, ask at most one follow-up question every 3 weeks.
If a customer is in STRATEGIC RISK, prioritize questions toward commercial counter-action, not explanation.
If a customer remains OPEN for more than 3 consecutive weeks, escalate from cause analysis to intervention recommendation.

You must explicitly apply this logic before selecting questions.

METADATA RULES (strict):
Every question must include all three fields in the metadata block:
(Distributor: <comma-separated>; Country: <comma-separated>; Customer: <comma-separated>)
Never leave a field blank.
If values are missing, infer using the provided data:
- Distributor and Country given, Customer missing: list all customers linked to that distributor in that country.
- Only Country given: list all distributors and customers in that country.
- Only Distributor given: list all countries and customers for that distributor.
- Customer missing or equal to Distributor: fill both fields with that value.
- If no valid value exists, use N/A.
Use semicolons between fields and commas within fields.
Use this exact metadata format for every question.
"""


response = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=900,          # <-- add this (700–1200 are all fine)
    temperature=0.2,         # optional but recommended for consistency
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt_text}
    ]
)
insights = response.choices[0].message.content.strip()
print("\n💡 GPT Insights:\n", insights)

def extract_questions_by_am(insights):
    """
    Corrected version of the original function. It reliably finds the 'Questions for...'
    header anywhere within an Account Manager's section of text.
    """
    am_sections = re.split(r"\n([A-Z][a-z]+ [A-Z][a-z]+)\n", "\n" + insights)
    am_data = []
    
    for i in range(1, len(am_sections), 2):
        am_name = am_sections[i].strip()
        section = am_sections[i + 1]
        
        # This is the corrected, simpler, and more robust pattern.
        # It finds "Questions for..." anywhere in the section and captures everything after it.
        q_match = re.search(
            r"Questions for.*:\s*\n(.*)", 
            section, 
            re.DOTALL | re.IGNORECASE
        )
        
        if not q_match:
            continue
            
        # q_match.group(1) now reliably contains the block of numbered questions
        questions_text = q_match.group(1)
        
        # This part correctly extracts each numbered question from the block
        for q_line in re.findall(r"\d+\.\s+(.+)", questions_text):
            am_data.append({
                "AM": am_name,
                "Question": q_line.strip()
            })
            
    return am_data

questions_by_am = extract_questions_by_am(insights)
new_rows = []
for q in questions_by_am:
    question_clean, distributors, countries, customers = extract_metadata_from_question(q['Question'])
    new_rows.append([week_num, year, q['AM'], ", ".join(distributors), ", ".join(countries), ", ".join(customers), question_clean, "", "Open", datetime.now().strftime("%Y-%m-%d")])

if new_rows:
    feedback_ws.append_rows(new_rows, value_input_option="USER_ENTERED")
    print(f"✅ Added {len(new_rows)} new questions to Feedback loop worksheet.")
else:
    print("No new questions found to add to the Feedback loop worksheet.")

# === Executive Summary Generation (with Product Trends) ===

def generate_product_trend_summary(recent_df, previous_df):
    """
    Calculates a stable 8-week vs. 8-week product trend and returns it as a text string.
    """
    try:
        recent_totals = recent_df.groupby('Product')['Total_mCi'].sum()
        previous_totals = previous_df.groupby('Product')['Total_mCi'].sum()
        combined = pd.DataFrame({'Recent 8 Weeks': recent_totals, 'Previous 8 Weeks': previous_totals}).fillna(0)
        
        calc_pct = lambda current, prev: ((current - prev) / prev * 100) if prev != 0 else np.inf
        combined['% Change'] = combined.apply(lambda row: calc_pct(row['Recent 8 Weeks'], row['Previous 8 Weeks']), axis=1)

        summary_lines = ["### PART 1: KEY PRODUCT TRENDS (8-Week vs. 8-Week) ###"]
        
        if combined.empty or (combined['Recent 8 Weeks'].sum() == 0 and combined['Previous 8 Weeks'].sum() == 0):
            summary_lines.append("No significant product sales activity was recorded in the last 16 weeks.")
            return "\n".join(summary_lines)

        for product, row in combined.iterrows():
            change = row['% Change']
            if change == np.inf:
                change_str = "which had no sales in the prior period."
            else:
                change_str = f"representing a change of {change:+.1f}%."
            
            summary_lines.append(f"- Sales for '{product}' in the last 8 weeks were {row['Recent 8 Weeks']:,.0f} mCi, {change_str}")
            
        return "\n".join(summary_lines)
    except Exception as e:
        return f"### PART 1: KEY PRODUCT TRENDS ###\nProduct-level trend data could not be generated: {e}"

# 1. Generate the stable product data as text
product_trend_summary = generate_product_trend_summary(recent_df, previous_df)

# 2. Define the new, highly-structured prompt
exec_summary_prompt ="""
You are a senior business analyst writing a high-level executive summary for company leadership. Your final output must be a **single, continuous block of text** with no paragraph breaks (no "\\n\\n").

**Instructions:**

1.  **Start with Product Trends:** Your first sentences MUST focus exclusively on the product-level performance outlined in 'PART 1: KEY PRODUCT TRENDS'. Summarize the most important overall product trends.

2.  **Transition to Customer Context:** Immediately after summarizing the product trends, use a transitional phrase like "This performance is reflected in customer-level activity..." or "Drilling down into customer data reveals..." and then provide context using the details from 'PART 2: CUSTOMER-LEVEL DATA'. Structure this analysis by first highlighting any significant **country-level or distributor-level patterns**, then mention key examples of increasing or decreasing **customers** that are driving the product trends.

**IMPORTANT RULES:**
- If 'PART 1' indicates no data is available, you MUST begin the summary directly with the customer data from 'PART 2'. Do not mention that product data is missing.
- Do NOT name Account Managers.
- Use clear, formal, business-oriented language.
- Your entire response MUST be a single block of text. Do not use bullets, bolding, or paragraph breaks.
"""

# 3. Combine the PRODUCT summary and the full DETAIL of the CUSTOMER summary
final_exec_prompt_content = (
    product_trend_summary +
    "\n\n-----\n\n" +
    "### PART 2: CUSTOMER-LEVEL DATA ###\n" +
    raw_report_text_for_exec_summary # Using the original, detailed customer report text
)

# 4. Diagnostic Print
print("\n" + "="*20 + " DEBUG: Final Content for Executive Summary " + "="*20)
print(final_exec_prompt_content)
print("="*75 + "\n")


# 5. Call the AI
exec_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": exec_summary_prompt},
        {"role": "user", "content": final_exec_prompt_content}
    ]
)
executive_summary = exec_response.choices[0].message.content.strip()
print("\n📝 Generated Executive Summary:\n", executive_summary)

summary_json_path = os.path.join(output_folder, "Executive_Summary_test.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump({"executive_summary": executive_summary}, f, ensure_ascii=False, indent=2)

upload_to_drive(summary_json_path, "Executive_Summary.json", folder_id)

# === Save Executive Summary also into Google Sheet ===
try:
    gsheet_service = build('sheets', 'v4', credentials=creds)

    # The spreadsheet ID for your target Google Sheet
    summary_spreadsheet_id = "185PnDw31D0zAYeUzNaSPQlQGTjmVVX1O9C86FC4JxV8"

    # Check if the sheet exists, if not create it
    sheet_metadata = gsheet_service.spreadsheets().get(spreadsheetId=summary_spreadsheet_id).execute()
    sheet_titles = [s['properties']['title'] for s in sheet_metadata['sheets']]

    if "Executive Summary" not in sheet_titles:
        requests_body = { "requests": [{ "addSheet": { "properties": { "title": "Executive Summary" } } }] }
        gsheet_service.spreadsheets().batchUpdate(spreadsheetId=summary_spreadsheet_id, body=requests_body).execute()
        print("✅ 'Executive Summary' sheet created.")

    # Clear existing content in cell A1 before writing new summary
    gsheet_service.spreadsheets().values().clear(
        spreadsheetId=summary_spreadsheet_id,
        range="Executive Summary!A1"
    ).execute()

    # Write the new executive summary into cell A1
    gsheet_service.spreadsheets().values().update(
        spreadsheetId=summary_spreadsheet_id,
        range="Executive Summary!A1",
        valueInputOption="RAW",
        body={ "values": [[executive_summary]] }
    ).execute()

    print("✅ Executive Summary successfully written to Google Sheet.")

except Exception as e:
    print(f"❌ Failed to write Executive Summary to Google Sheet: {e}")

with open(insight_history_path, "a") as f:
    f.write(f"\n\n===== Week {week_num}, {year} =====\n{insights}")

# === PDF Generation ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}_test.pdf")
with PdfPages(latest_pdf) as pdf:
    print("DEBUG: Entered PdfPages block")
    if not any([stopped, decreased, increased, inactive_recent_4]):
        fig = plt.figure(figsize=(8.5, 11)); plt.axis("off")
        fig.text(0.5, 0.5, "No data available for this week's report.", ha="center", va="center", fontsize=18)
        pdf.savefig(fig); plt.close(fig)
    else:
        # --- Cover Page ---
        fig = plt.figure(figsize=(9.5, 11)); plt.axis("off")
        fig.text(0.5, 0.80, "Weekly Orders Report", fontsize=26, ha="center", weight='bold')
        fig.text(0.5, 0.74, f"Week {week_num}, {year}", fontsize=20, ha="center")
        logo_path = os.path.join(script_dir, "Isotopia.jpg")
        if os.path.exists(logo_path):
            logo = mpimg.imread(logo_path)
            ax_logo = fig.add_axes([(1 - 0.35) / 2, 0.40, 0.35, 0.18])
            ax_logo.imshow(logo); ax_logo.axis("off")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # --- SECOND PAGE: Product Comparison Tables ---
        add_comparison_tables_page_to_pdf(pdf, df)
        
        # --- Summary Table: Overall Categories ---
        total_decreased = sum(curr - prev for v in decreased_by_am.values() for _, prev, curr, _ in v)
        total_increased = sum(curr - prev for v in increased_by_am.values() for _, prev, curr, _ in v)
        summary_df = pd.DataFrame([["Stopped Ordering", len(stopped), "–"],["Decreased Orders", sum(len(v) for v in decreased_by_am.values()), f"{total_decreased:+.1f}"],["Increased Orders", sum(len(v) for v in increased_by_am.values()), f"{total_increased:+.1f}"],["Inactive (last 4 wks)", len(inactive_recent_4), "–"]], columns=["Category", "# of Customers", "Total Δ (mCi)"])
        fig, ax = plt.subplots(figsize=(8.5, max(2.2, 0.4 + 0.3 * len(summary_df)))); ax.axis("off")
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.1, 1.4)
        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.2
            if row == 0: cell.set_facecolor("#e6e6fa"); cell.set_text_props(weight='bold', ha="center")
            else: cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff"); cell.set_text_props(ha="center")
        ax.set_title("Summary by Category", fontsize=14, weight='bold', pad=10)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # --- Summary Table: By Account Manager ---
        am_list = set(list(decreased_by_am.keys()) + list(increased_by_am.keys()) + [customer_to_manager.get(name, "Other") for name in inactive_recent_4 + [x[0] for x in stopped]])
        am_summary = [[am, len(increased_by_am.get(am, [])), len(decreased_by_am.get(am, [])), sum(1 for name in inactive_recent_4 if customer_to_manager.get(name, "Other") == am), sum(1 for name, _ in stopped if customer_to_manager.get(name, "Other") == am)] for am in sorted(am_list)]
        am_df = pd.DataFrame(am_summary, columns=["Account Manager", "+ Customers", "– Customers", "Inactive", "Stopped"])
        fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.4 + 0.3 * len(am_df)))); ax.axis("off")
        table = ax.table(cellText=am_df.values, colLabels=am_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.1, 1.4)
        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.2
            if row == 0: cell.set_facecolor("#e6e6fa"); cell.set_text_props(weight='bold', ha="center")
            else: cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff"); cell.set_text_props(ha="center")
        ax.set_title("Summary by Account Manager", fontsize=14, weight='bold', pad=10)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        
        # --- Top 5 NCA Distributors (NEW) ---
        nca_product_name = "Lutetium  (177Lu) chloride N.C.A."
        nca_df = recent_df[recent_df["Product"] == nca_product_name]

        if not nca_df.empty:
            # Aggregate by Distributor
            top_distributors = (
                nca_df.groupby("Distributor")["Total_mCi"]
                .sum()
                .nlargest(5)
                .index
            )

            dist_plot_df = nca_df[nca_df["Distributor"].isin(top_distributors)].copy()
            if not dist_plot_df.empty:
                dist_plot_df["WeekLabel"] = (
                    dist_plot_df["Year"].astype(str)
                    + "-W"
                    + dist_plot_df["Week"].astype(str).str.zfill(2)
                )

                pivot_dist = (
                    dist_plot_df.pivot_table(
                        index="WeekLabel",
                        columns="Distributor",
                        values="Total_mCi",
                        aggfunc="sum",
                    )
                    .fillna(0)
                )

                # Sort weeks chronologically
                pivot_dist = pivot_dist.reindex(
                    sorted(
                        pivot_dist.index,
                        key=lambda x: (int(x.split("-W")[0]), int(x.split("-W")[1])),
                    )
                )

                fig, ax = plt.subplots(figsize=(8, 4.5))
                pivot_dist.plot(ax=ax, marker="o")
                ax.set_title("Top 5 N.C.A. Distributors", fontsize=16, weight="bold")
                ax.set_xlabel("Week of Supply", fontsize=11)
                ax.set_ylabel("Total mCi Ordered", fontsize=11)
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, linestyle="--", alpha=0.5)

                handles, labels = ax.get_legend_handles_labels()
                wrapped_labels = ["\n".join(textwrap.wrap(l, 25)) for l in labels]
                ax.legend(
                    handles,
                    wrapped_labels,
                    title="Distributor",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    fontsize=8,
                )

                fig.tight_layout(pad=2.0)
                pdf.savefig(fig)
                plt.close(fig)
 
        # --- Top 5 Charts by Product ---
        products = {"Lutetium  (177Lu) chloride N.C.A.": "Top 5 N.C.A. Customers", "Lutetium (177Lu) chloride C.A": "Top 5 C.A. Customers", "Terbium-161 chloride n.c.a": "Top 5 Terbium Customers"}
        for product_name, title in products.items():
            product_df = recent_df[recent_df["Product"] == product_name]
            if product_df.empty: continue
            top_customers = product_df.groupby("Customer")["Total_mCi"].sum().nlargest(5).index
            plot_df = product_df[product_df["Customer"].isin(top_customers)].copy()
            plot_df["WeekLabel"] = plot_df["Year"].astype(str) + "-W" + plot_df["Week"].astype(str).str.zfill(2)
            pivot_df = plot_df.pivot_table(index="WeekLabel", columns="Customer", values="Total_mCi", aggfunc="sum").fillna(0)
            pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=lambda x: (int(x.split("-W")[0]), int(x.split("-W")[1]))))
            fig, ax = plt.subplots(figsize=(8, 4.5)); pivot_df.plot(ax=ax, marker='o')
            ax.set_title(title, fontsize=16, weight='bold'); ax.set_xlabel("Week of Supply", fontsize=11); ax.set_ylabel("Total mCi Ordered", fontsize=11)
            ax.tick_params(axis='x', rotation=45); ax.grid(True, linestyle='--', alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            wrapped_labels = ['\n'.join(textwrap.wrap(l, 25)) for l in labels]
            ax.legend(handles, wrapped_labels, title="Customer", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            fig.tight_layout(pad=2.0); pdf.savefig(fig); plt.close(fig)

        # --- Customer Trend Tables ---
        if stopped:
            stopped_df = pd.DataFrame([[name, wrap_text(customer_to_manager.get(name, "Other"))] for name, _ in sorted(stopped, key=lambda x: customer_to_manager.get(x[0], "Other"))], columns=["Customer", "Account Manager"])
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(stopped_df)) + 1)); ax.axis("off")
            table = ax.table(cellText=stopped_df.values, colLabels=stopped_df.columns, loc="upper left", cellLoc="left", colWidths=[0.8, 0.2])
            table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left")
            ax.set_title("STOPPED ORDERING", fontsize=14, weight="bold", pad=15); fig.text(0.5, 0.87, "Customers who stopped ordering in the last 4 weeks...", fontsize=10, ha="center")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        if decreased:
            for am, am_rows in sorted(decreased_by_am.items()):
                decreased_df = pd.DataFrame([[name, f"{curr - prev:+.0f}", f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%", wrap_text(am)] for name, prev, curr, am in sorted(am_rows, key=lambda x: (x[2]-x[1])/(x[1] if x[1]!=0 else 1))], columns=["Customer", "Change (mCi)", "% Change", "Account Manager"])
                fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(decreased_df)) + 1)); ax.axis("off")
                table = ax.table(cellText=decreased_df.values, colLabels=["Customer", "Change (mCi)", "% Change", "Account Manager"], loc="upper left", cellLoc="center", colWidths=[0.64, 0.11, 0.11, 0.14])
                table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                    cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left" if col == 0 else "center")
                ax.set_title(f"DECREASED ORDERS – {am}", fontsize=14, weight="bold", pad=15); fig.text(0.5, 0.87, f"Customers managed by {am} who decreased orders...", fontsize=10, ha="center")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        if increased:
            for am, am_rows in sorted(increased_by_am.items()):
                increased_df = pd.DataFrame([[name, f"{curr - prev:+.0f}", f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%", wrap_text(am)] for name, prev, curr, am in sorted(am_rows, key=lambda x: (x[2]-x[1])/(x[1] if x[1]!=0 else 1), reverse=True)], columns=["Customer", "Change (mCi)", "% Change", "Account Manager"])
                fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(increased_df)) + 1)); ax.axis("off")
                table = ax.table(cellText=increased_df.values, colLabels=["Customer", "Change (mCi)", "% Change", "Account Manager"], loc="upper left", cellLoc="center", colWidths=[0.64, 0.11, 0.11, 0.14])
                table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                    cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left" if col == 0 else "center")
                ax.set_title(f"INCREASED ORDERS – {am}", fontsize=14, weight="bold", pad=15); fig.text(0.5, 0.87, f"Customers managed by {am} who increased orders...", fontsize=10, ha="center")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        if inactive_recent_4:
            inactive_df = pd.DataFrame([[name, wrap_text(customer_to_manager.get(name, "Other"))] for name in sorted(inactive_recent_4, key=lambda name: customer_to_manager.get(name, "Other"))], columns=["Customer", "Account Manager"])
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(inactive_df)) + 1)); ax.axis("off")
            table = ax.table(cellText=inactive_df.values, colLabels=inactive_df.columns, loc="upper left", cellLoc="left", colWidths=[0.8, 0.2])
            table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left")
            ax.set_title("INACTIVE IN PAST 4 WEEKS", fontsize=14, weight="bold", pad=15); fig.text(0.5, 0.87, "Customers who were active 4–8 weeks ago...", fontsize=10, ha="center")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

# --- ChatGPT Insights Pages ---
        insight_lines = [extract_metadata_from_question(line)[0] for line in insights.split("\n")]
        
        # This logic handles line wrapping and preserves blank lines for paragraph spacing
        wrapped_insights = []
        for line in insight_lines:
            if not line.strip():
                wrapped_insights.append("") # Keep the blank line
            else:
                wrapped_insights.extend(textwrap.wrap(line, width=100, break_long_words=False))

        lines_per_page = 30  # Reduced from 35 to account for more spacing
        
        if not wrapped_insights:
            fig = plt.figure(figsize=(9.5, 11)); plt.axis("off")
            fig.text(0.5, 0.5, "No insights available this week.", ha="center", fontsize=14)
            pdf.savefig(fig); plt.close(fig)
        else:
            for page_start in range(0, len(wrapped_insights), lines_per_page):
                fig = plt.figure(figsize=(9.5, 11)); plt.axis("off")
                page_lines = wrapped_insights[page_start:page_start + lines_per_page]
                
                # Use a line-by-line drawing method to control spacing
                y_cursor = 0.95  # Start near the top of the page
                line_spacing = 0.033 # Increased from ~0.028 for more space

                for line in page_lines:
                    fig.text(0.06, y_cursor, line, fontsize=10, ha="left", va="top")
                    y_cursor -= line_spacing # Move the y-cursor down for the next line

                pdf.savefig(fig)
                plt.close(fig)

# === Save report with additional filenames ===
summary_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Summary_Week_{week_num}_{year}.pdf")
latest_copy_path = os.path.join(output_folder, "Latest_Weekly_Report.pdf")
copyfile(latest_pdf, summary_pdf)
copyfile(latest_pdf, latest_copy_path)
print(f"✅ Report also saved as: {summary_pdf}")
print(f"✅ Report also saved as: {latest_copy_path}")

# === Add a short pause to ensure files are fully written and closed ===
import time
time.sleep(0.5)

# === Create and save week info file ===
week_info_path = os.path.join(output_folder, "Week_number.txt")
with open(week_info_path, "w") as f:
    f.write(f"{week_num},{year}")

# === Upload PDFs to Google Drive Folder ===
upload_to_drive(summary_pdf, f"Weekly_Orders_Report_Summary_Week_{week_num}_{year}.pdf", folder_id)
upload_to_drive(latest_copy_path, "Latest_Weekly_Report.pdf", folder_id)
upload_to_drive(week_info_path, f"Week_number.txt", folder_id)















