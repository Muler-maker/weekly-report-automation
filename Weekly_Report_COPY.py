# === Imports ===
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os, re
from shutil import copyfile
import textwrap
from email.message import EmailMessage
from openai import OpenAI
import requests
import base64
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import json

def wrap_text(text, width=20):
    return '\n'.join(textwrap.wrap(str(text), width=width))

def build_feedback_context(feedback_df, week_num, year):
    context_lines = []
    for idx, row in feedback_df.iterrows():
        if (
            (str(row['Status']).lower() != 'done') or
            (str(row['Week']) == str(week_num) and str(row['Year']) == str(year))
        ):
            line = (
                f"Account Manager: {row['AM']}\n"
                f"Distributor: {row['Distributor']} | Country: {row['Country/Countries']} | Customers: {row['Customers']}\n"
                f"Last question: {row['Question']}\n"
                f"AM answer: {row['Comments / Feedback']}\n"  # or whatever your actual column name is
                f"Status: {row['Status']}, Date: {row['Feedback Date']}\n"
                "------"
            )
            context_lines.append(line)
    return "\n".join(context_lines)

# --- Your utility/setup variables here ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1uJAArBkHXgFvY_JPIieKRjiOYd9-Ys7Ii9nXT3fWbUg"

# === Authenticate Google Sheets (do this before loading sheets) ===
SERVICE_ACCOUNT_FILE = os.path.join(script_dir, 'credentials.json')
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)

# === Now load the Feedback loop worksheet ===
feedback_ws = gc.open_by_url(SPREADSHEET_URL).worksheet("Feedback loop")
feedback_df = pd.DataFrame(feedback_ws.get_all_records())
feedback_df.columns = [c.strip() for c in feedback_df.columns]


# === Set global font style ===
plt.rcParams["font.family"] = "DejaVu Sans"

# === Set local output folder ===
output_folder = os.path.join(os.getcwd(), 'Reports')
os.makedirs(output_folder, exist_ok=True)

# === Define target Google Drive folder ID ===
folder_id = "1i1DAOTnF8SznikYrS-ovrg2TRgth9wwP"

# === Google Sheets Service Account Authentication ===
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
SERVICE_ACCOUNT_FILE = os.path.join(script_dir, 'credentials.json')
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)

# === Load data from Google Sheets ===
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1WNW4IqAG6W_6XVSaSk9kkukztY93qCDQ3mypZaF5F-Y"
sheet = gc.open_by_url(SPREADSHEET_URL).worksheet("Airtable Data")
df = pd.DataFrame(sheet.get_all_records())
df.columns = [c.strip() for c in df.columns]

# === Add Account Manager mapping ===
account_manager_map = {
    "vbeillis@isotopia-global.com": "Vicki Beillis",
    "naricha@isotopia-global.com": "Noam Aricha",
    "ndellus@isotopia-global.com": "Noam Dellus",
    "gbader@isotopia-global.com": "Gilli Bader"
}
df["Account Manager"] = df["Account Manager Email"].map(account_manager_map).fillna("Other")

# === Define email sending function using Resend API ===
def send_email(subject, body, to_emails, attachment_path):
    if not os.path.exists(attachment_path):
        print(f"❌ Attachment file not found: {attachment_path}")
        return

    with open(attachment_path, "rb") as f:
        files = {
            'attachments': (os.path.basename(attachment_path), f, 'application/pdf')
        }

        data = {
            "from": "Dan Amit <dan.test@resend.dev>",
            "to": ','.join(to_emails),  # Must be a comma-separated string
            "subject": subject,
            "text": body
        }

        headers = {
            "Authorization": f"Bearer {os.getenv('RESEND_API_KEY')}"
            # Do NOT add "Content-Type" — requests sets it automatically
        }

        response = requests.post("https://api.resend.com/emails", data=data, files=files, headers=headers)

    if 200 <= response.status_code < 300:
        print(f"📨 Email successfully sent to: {', '.join(to_emails)}")
    else:
        print("❌ Failed to send email:", response.status_code, response.text)

def upload_to_drive(file_path, file_name, folder_id=None):
    drive_service = build('drive', 'v3', credentials=creds)

    # Search for existing files with the same name in the folder
    query = f"name='{file_name}' and trashed=false"
    if folder_id:
        query += f" and '{folder_id}' in parents"

    existing_files = drive_service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute().get('files', [])
    
    # Delete existing files
    for file in existing_files:
        drive_service.files().delete(fileId=file['id']).execute()
        print(f"🗑️ Deleted: {file['name']} (ID: {file['id']})")

    # Upload the new file
    file_metadata = {
        'name': file_name,
        'parents': [folder_id] if folder_id else []
    }

    mimetype = 'application/json' if file_name.endswith('.json') else 'application/pdf'
    media = MediaFileUpload(file_path, mimetype=mimetype)
    uploaded_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    print(f"✅ Uploaded: {file_name} (ID: {uploaded_file.get('id')})")

# === Define report date values (simulate 11 days ahead) ===
today = datetime.today() + timedelta(days=11)
week_num = today.isocalendar().week
year = today.isocalendar().year

# === Prepare weekly report data ===
df = df.rename(columns={
    "The customer": "Customer",
    "Total amount ordered (mCi)": "Total_mCi",
    "Year": "Year",
    "Week of supply": "Week",
    "Shipping Status": "ShippingStatus",
    "Catalogue description (sold as)": "Product",
    "Distributing company (from Company name)": "Distributor",
    "Country": "Country"
})

valid_statuses = ["Shipped", "Partially shipped", "Shipped and arrived late", "Order being processed"]
df = df[df["ShippingStatus"].isin(valid_statuses)]
df = df.dropna(subset=["Customer", "Total_mCi", "Year", "Week", "ShippingStatus", "Product"])
df["Total_mCi"] = pd.to_numeric(df["Total_mCi"], errors="coerce")
df = df[pd.to_numeric(df["Year"], errors="coerce").notnull()]
df["Year"] = df["Year"].astype(int)
df = df[pd.to_numeric(df["Week"], errors="coerce").notnull()]
df["Week"] = df["Week"].astype(int)
df = df.dropna(subset=["Total_mCi", "Year", "Week"])
df["YearWeek"] = list(zip(df["Year"], df["Week"]))

# === Define 16-week window ===
current = today - timedelta(days=today.weekday())
week_pairs = [(d.isocalendar().year, d.isocalendar().week) for d in [current - timedelta(weeks=(15 - i)) for i in range(16)]]
previous_8, recent_8 = week_pairs[:8], week_pairs[8:]
recent_df = df[df["YearWeek"].isin(recent_8)]
previous_df = df[df["YearWeek"].isin(previous_8)]

# === Trends Analysis ===
recent_totals = recent_df.groupby("Customer")["Total_mCi"].sum()
previous_totals = previous_df.groupby("Customer")["Total_mCi"].sum()
customer_to_manager = df.set_index("Customer")["Account Manager"].to_dict()

all_customers = set(df["Customer"])
increased, decreased, stopped = [], [], []
for customer in all_customers:
    prev = previous_totals.get(customer, 0)
    curr = recent_totals.get(customer, 0)
    manager = customer_to_manager.get(customer, "Other")
    if prev == 0 and curr > 0:
        increased.append((customer, prev, curr, manager))
    elif curr == 0 and prev > 0:
        stopped.append((customer, manager))
    elif curr > prev:
        increased.append((customer, prev, curr, manager))
    elif curr < prev:
        decreased.append((customer, prev, curr, manager))

# === Build mapping: customer to distributor ===
customer_to_distributor = df.set_index("Customer")["Distributor"].to_dict()
customer_to_country = df.set_index("Customer")["Country"].to_dict()

# === For GPT prompt only: create formatted lines with distributor info ===
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
gpt_distributor_section = [
    "Customer name                     | Change   | % Change | Account Manager | Distributor | Country"
] + gpt_rows

# === Inactive recent 4 weeks ===
last_8_weeks = week_pairs[-8:]
previous_4_weeks, recent_4_weeks = last_8_weeks[:4], last_8_weeks[4:]
active_previous_4 = set(df[df["YearWeek"].isin(previous_4_weeks)]["Customer"])
active_recent_4 = set(df[df["YearWeek"].isin(recent_4_weeks)]["Customer"])
inactive_recent_4 = sorted(active_previous_4 - active_recent_4)

# === Format summary text ===
format_row = lambda name, prev, curr, mgr: (
    f"{name:<30} | {curr - prev:+7.0f} mCi | {(curr - prev) / prev * 100 if prev else 100:+6.1f}% | {mgr}",
    (curr - prev) / prev * 100 if prev else 100
)
increased_formatted = sorted([format_row(*x) for x in increased], key=lambda x: x[1], reverse=True)
decreased_formatted = sorted([format_row(*x) for x in decreased], key=lambda x: x[1])

summary_lines = [
    "STOPPED ORDERING:",
    "Customers who stopped ordering in the last 4 weeks but did order in the 4 weeks before.",
    "Customer Name                     | Account Manager",
    "---------------------------------------------------"
] + [f"{x[0]:<35} | {x[1]}" for x in stopped] if stopped else ["- None"]

summary_lines += [
    "",
    "DECREASED ORDERS:",
    "These customers ordered less in the last 8 weeks compared to the 8 weeks prior.",
    "Customer Name                     | Change   | % Change | Account Manager",
    "--------------------------------------------------------------------------"
] + [x[0] for x in decreased_formatted] if decreased_formatted else ["- None"]

summary_lines += [
    "",
    "INCREASED ORDERS:",
    "These customers increased their order amounts in the last 8 weeks compared to the 8 weeks prior.",
    "Customer Name                     | Change   | % Change | Account Manager",
    "--------------------------------------------------------------------------"
] + [x[0] for x in increased_formatted] if increased_formatted else ["- None"]

summary_lines += [
    "",
    "INACTIVE IN PAST 4 WEEKS:",
    "Customers who ordered in the previous 4 weeks but not in the most recent 4.",
    "Customer Name",
    "-------------------------------"
] + inactive_recent_4 if inactive_recent_4 else ["- None"]

# === Save summary ===
summary_path = os.path.join(output_folder, "summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

# === ChatGPT Insights with memory ===

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=openai_api_key)

# Read summary report
with open(summary_path, "r", encoding="utf-8") as f:
    report_text = f.read()

# Optional: Prepend past insights for context/memory
insight_history_path = os.path.join(output_folder, "insight_history.txt")
if os.path.exists(insight_history_path):
    with open(insight_history_path, "r", encoding="utf-8") as f:
        past_insights = f.read()
    # Combine with a clear marker
    report_text = f"{past_insights}\n\n===== NEW WEEK =====\n\n{report_text}"

# === Build feedback context from the Feedback loop sheet ===
feedback_context = build_feedback_context(feedback_df, week_num, year)
if not feedback_context.strip():
    feedback_context = "No previous feedback available for this period."

# === Combine feedback, summary report, and distributor info for ChatGPT ===
full_prompt_text = (
    "Previous feedback from Account Managers (from the Feedback loop sheet):\n"
    + feedback_context +
    "\n\n==== New Weekly Report Data ====\n\n"
    + report_text +
    "\n\n=== Distributor info for GPT analysis ===\n"
    + "\n".join(gpt_distributor_section)
)

# System prompt with analyst instructions
system_prompt = system_prompt = """
You are a senior business analyst. Analyze the weekly report for each Account Manager using a top-down structure: Distributor-level trends, followed by country-level patterns, and then customer-specific insights.

For each Account Manager:
- Start their section with their full name on a line by itself (e.g., Vicki Beillis).
- If there are no significant trends or issues for that week, write: No significant insights or questions for this week.
- Otherwise, provide a concise summary of relevant trends and insights, grouped as follows:
    Distributor-level: Note important patterns, risks, or deviations, including expected order behaviors.
    Country-level: Summarize trends affecting several customers in the same country.
    Customer-specific: Highlight notable changes, spikes, drops, or inactivity at the customer level.

After the insights for each Account Manager, always include a separate section titled exactly as follows:
Questions for [Account Manager Name]:
1. [First question]
2. [Second question]
3. [Third question]
(Use numbered questions, one per line, and use the AM’s exact name in the heading.)

Guidelines:
- Base your questions on both the current report and the most recent feedback or answers from previous cycles.
- For ongoing or unresolved issues, ask clarifying or follow-up questions and reference the previous feedback where relevant.
- For new issues, ask investigative questions to help clarify the root cause or suggest possible next steps.
- Reference previous reports and feedback to highlight new, ongoing, or resolved issues.
- Present only insights, trends, and questions—do not include recommendations or action items.
- Use only plain text. Do not use Markdown, asterisks, or special formatting of any kind.
- COMISSÃO NACIONAL DE ENERGIA NUCLEAR (CNEN) is expected to order every two weeks on even-numbered weeks. Flag and ask about any deviation from this pattern.
"""
# Call OpenAI chat completion
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt_text}
    ]
)

insights = response.choices[0].message.content.strip()
print("\n💡 GPT Insights:\n", insights)

# === ADD THIS BLOCK HERE ===

import re
from datetime import datetime

def extract_questions_by_am(insights):
    am_sections = re.split(r"\n([A-Z][a-z]+ [A-Z][a-z]+)\n", "\n" + insights)
    am_data = []
    for i in range(1, len(am_sections), 2):
        am_name = am_sections[i].strip()
        section = am_sections[i + 1]
        q_match = re.search(r"Questions\s*(for\s*[A-Za-z ]+)?\s*:\s*(.+?)(?=(\n[A-Z][a-z]+ [A-Z][a-z]+\n|$))", section, re.DOTALL)
        if not q_match:
            continue
        questions_text = q_match.group(2)
        for q_line in re.findall(r"\d+\.\s+(.+)", questions_text):
            am_data.append({
                "AM": am_name,
                "Question": q_line.strip()
            })
    return am_data

questions_by_am = extract_questions_by_am(insights)

# Get all customer names, sorted longest first to avoid substring issues
customer_names = sorted(df["Customer"].dropna().unique(), key=lambda x: -len(x))

def extract_customers_from_question(question, customer_names):
    found = []
    q_lower = question.lower()
    for cname in customer_names:
        if cname.lower() in q_lower:
            found.append(cname)
    return list(set(found))  # Unique list, in case of repeats

new_rows = []
for q in questions_by_am:
    customer_list = extract_customers_from_question(q['Question'], customer_names)
    if customer_list:
        relevant_distributors = set()
        relevant_countries = set()
        for customer in customer_list:
            subdf = df[df["Customer"] == customer]
            relevant_distributors.update(subdf["Distributor"].dropna().unique())
            relevant_countries.update(subdf["Country"].dropna().unique())
        distributors = ", ".join(sorted(relevant_distributors))
        countries = ", ".join(sorted(relevant_countries))
        customers = ", ".join(sorted(customer_list))
    else:
        distributors = ""
        countries = ""
        customers = ""
    new_rows.append([
        week_num,
        year,
        q['AM'],
        distributors,
        countries,
        customers,
        q['Question'],
        "",  # Comments/Feedback
        "Open",
        datetime.now().strftime("%Y-%m-%d")
    ])

if new_rows:
    feedback_ws.append_rows(new_rows, value_input_option="USER_ENTERED")
    print(f"✅ Added {len(new_rows)} new questions to Feedback loop worksheet.")
else:
    print("No new questions found to add to the Feedback loop worksheet.")

# === Generate Executive Summary ===
exec_summary_prompt ="""
You are a senior business analyst. Based on the report below, write a very short executive summary in the form of one or two concise paragraphs, suitable for company leadership.

Guidelines:
- Do not include any greeting or opening line
- Focus on major trends across countries and distributors
- You may name specific customers if they help illustrate a broader trend
- Do not mention Account Managers
- Use clear, business-oriented language
- Space the text into one or two short paragraphs for readability
- Avoid technical jargon or formatting (no bullets, bold, etc.)
- Use an approcheable and simple language
"""

exec_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": exec_summary_prompt},
        {"role": "user", "content": report_text}
    ]
)
executive_summary = exec_response.choices[0].message.content.strip()

# Save executive summary to JSON
summary_json_path = os.path.join(output_folder, "Executive_Summary.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "executive_summary": executive_summary
    }, f, ensure_ascii=False, indent=2)

# Upload JSON to Google Drive (replacing existing file if any)
upload_to_drive(summary_json_path, "Feedback_Executive_Summary.json", folder_id)

# === Save new insight to history ===
with open(insight_history_path, "a") as f:
    f.write(f"\n\n===== Week {week_num}, {year} =====\n")
    f.write(insights)

# === PDF Generation and email sending follow ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}.pdf")

with PdfPages(latest_pdf) as pdf:
    print("DEBUG: Entered PdfPages block")

    if not any([stopped, decreased, increased, inactive_recent_4]):
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        fig.text(
            0.5, 0.5,
            "No data available for this week's report.",
            ha="center", va="center", fontsize=18
        )
        pdf.savefig(fig)
        print("⚠️ No data available. Generated fallback PDF with message page only.")
        plt.close(fig)
    else:
        # --- Cover Page ---
        fig = plt.figure(figsize=(9.5, 11))
        plt.axis("off")

        fig.text(0.5, 0.80, "Weekly Orders Report", fontsize=26, ha="center", va="center", weight='bold')
        fig.text(0.5, 0.74, f"Week {week_num}, {year}", fontsize=20, ha="center", va="center")

        logo_path = os.path.join(script_dir, "Isotopia.jpg")
        if os.path.exists(logo_path):
            logo = mpimg.imread(logo_path)
            logo_width = 0.35
            logo_height = 0.18
            logo_x = (1 - logo_width) / 2
            logo_y = 0.40
            ax_logo = fig.add_axes([logo_x, logo_y, logo_width, logo_height])
            ax_logo.imshow(logo)
            ax_logo.axis("off")
        else:
            print("WARNING: Logo file not found, skipping logo.")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- STOPPED ORDERING TABLE ---
        if stopped:
            stopped_df = pd.DataFrame(
                [
                    [name, wrap_text(mgr)]
                    for name, mgr in stopped
                ],
                columns=["Customer", "Account Manager"]
            )
            fig_height = max(4.5, 0.4 + 0.3 * len(stopped_df))
            fig, ax = plt.subplots(figsize=(11, fig_height + 1))
            ax.axis("off")
            table = ax.table(
                cellText=stopped_df.values,
                colLabels=stopped_df.columns,
                loc="upper left",
                cellLoc="left",
                colWidths=[0.8, 0.2]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1.0, 1.4)

            # Improve alignment and styling
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2
                if row == 0:
                    cell.set_facecolor("#e6e6fa")
                    if col == 0:
                        cell.set_text_props(ha="left", weight='bold')
                    else:
                        cell.set_text_props(ha="left", weight='bold')
                else:
                    if row % 2 == 0:
                        cell.set_facecolor("#f9f9f9")
                    else:
                        cell.set_facecolor("#ffffff")
                    if col == 0:
                        cell.set_text_props(ha="left")
                        cell._loc = 'left'
                    else:
                        cell.set_text_props(ha="left")
                        cell._loc = 'left'

            ax.set_title("STOPPED ORDERING", fontsize=14, weight="bold", pad=15)
            fig.text(
                0.5, 0.87,
                "Customers who stopped ordering in the last 4 weeks but did order in the 4 weeks before.",
                fontsize=10, ha="center"
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

                   # --- DECREASED ORDERS TABLE ---
        if decreased:
            decreased_df = pd.DataFrame([
                [
                    name,
                    f"{curr - prev:+.0f}",
                    f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%",
                    wrap_text(mgr),
                    (curr - prev) / prev * 100 if prev else 100  # Helper column
                ]
                for name, prev, curr, mgr in decreased
            ], columns=["Customer", "Change (mCi)", "% Change", "Account Manager", "PercentValue"])
            # Sort by PercentValue ascending (worst decreases at top)
            decreased_df = decreased_df.sort_values("PercentValue", ascending=True).drop(columns="PercentValue")
            fig_height = max(4.5, 0.4 + 0.3 * len(decreased_df))
            fig, ax = plt.subplots(figsize=(11, fig_height + 1))
            ax.axis("off")
            # Set colWidths: [Customer, Change, % Change, Account Manager]
            colWidths = [0.64, 0.11, 0.11, 0.14]
            table = ax.table(
                cellText=decreased_df.values,
                colLabels=decreased_df.columns,
                loc="upper left",
                cellLoc="center",    # Default: center for all cells, will override Customer later
                colWidths=colWidths
            )
            # Set larger font size for table
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2
                # Header row formatting
                if row == 0:
                    cell.set_facecolor("#e6e6fa")
                    cell.set_text_props(weight='bold')
                    # Align "Customer" header left, rest center
                    if col == 0:
                        cell.set_text_props(ha="left")
                        cell._loc = 'left'
                    else:
                        cell.set_text_props(ha="center")
                        cell._loc = 'center'
                else:
                    # Stripe effect
                    if row % 2 == 0:
                        cell.set_facecolor("#f9f9f9")
                    else:
                        cell.set_facecolor("#ffffff")
                    # Align "Customer" column left, others center
                    if col == 0:
                        cell.set_text_props(ha="left")
                        cell._loc = 'left'
                    else:
                        cell.set_text_props(ha="center")
                        cell._loc = 'center'
            ax.set_title("DECREASED ORDERS", fontsize=16, weight="bold", pad=15)
            fig.text(
                0.5, 0.87,
                "These customers ordered less in the last 8 weeks compared to the 8 weeks prior.",
                fontsize=12, ha="center"
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)




          # --- INCREASED ORDERS TABLE ---
        if increased:
            increased_df = pd.DataFrame([
                [
                    name,
                    f"{curr - prev:+.0f}",
                    f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%",
                    wrap_text(mgr),
                    (curr - prev) / prev * 100 if prev else 100  # Helper column
                ]
                for name, prev, curr, mgr in increased
            ], columns=["Customer", "Change (mCi)", "% Change", "Account Manager", "PercentValue"])
            # Sort by PercentValue descending (biggest increases at top)
            increased_df = increased_df.sort_values("PercentValue", ascending=False).drop(columns="PercentValue")
            fig_height = max(4.5, 0.4 + 0.3 * len(increased_df))
            fig, ax = plt.subplots(figsize=(11, fig_height + 1))
            ax.axis("off")
            colWidths = [0.64, 0.11, 0.11, 0.14]
            table = ax.table(
                cellText=increased_df.values,
                colLabels=increased_df.columns,
                loc="upper left",
                cellLoc="center",   # Default: center for all cells, override for Customer column
                colWidths=colWidths
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2
                # Header row formatting
                if row == 0:
                    cell.set_facecolor("#e6e6fa")
                    cell.set_text_props(weight='bold')
                    if col == 0:
                        cell.set_text_props(ha="left")
                        cell._loc = 'left'
                    else:
                        cell.set_text_props(ha="center")
                        cell._loc = 'center'
                else:
                    if row % 2 == 0:
                        cell.set_facecolor("#f9f9f9")
                    else:
                        cell.set_facecolor("#ffffff")
                    if col == 0:
                        cell.set_text_props(ha="left")
                        cell._loc = 'left'
                    else:
                        cell.set_text_props(ha="center")
                        cell._loc = 'center'
            ax.set_title("INCREASED ORDERS", fontsize=16, weight="bold", pad=15)
            fig.text(
                0.5, 0.87,
                "These customers increased their order amounts in the last 8 weeks compared to the 8 weeks prior.",
                fontsize=12, ha="center"
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


        # --- INACTIVE IN PAST 4 WEEKS TABLE ---
        if inactive_recent_4:
            inactive_df = pd.DataFrame(
                [[name] for name in inactive_recent_4],
                columns=["Customer"]
            )
            fig_height = max(4.5, 0.4 + 0.3 * len(inactive_df))
            fig, ax = plt.subplots(figsize=(11, fig_height + 1))
            ax.axis("off")
            table = ax.table(
                cellText=inactive_df.values,
                colLabels=inactive_df.columns,
                loc="upper left",
                cellLoc="left",
                colWidths=[0.9]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2
                if row == 0:
                    cell.set_facecolor("#e6e6fa")
                    cell.set_text_props(ha="left", weight='bold')
                elif row % 2 == 0:
                    cell.set_facecolor("#f9f9f9")
                    cell.set_text_props(ha="left")
                    cell._loc = 'left'
                else:
                    cell.set_facecolor("#ffffff")
                    cell.set_text_props(ha="left")
                    cell._loc = 'left'
            ax.set_title("INACTIVE IN PAST 4 WEEKS", fontsize=14, weight="bold", pad=15)
            fig.text(
                0.5, 0.87,
                "Customers who were active 4–8 weeks ago but not in the most recent 4 weeks.",
                fontsize=10, ha="center"
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # === Add ChatGPT insights pages (paginated) ===
    insight_lines = insights.split("\n")
    wrapped_insights = []
    for line in insight_lines:
        wrapped_insights.extend(
            textwrap.wrap(line, width=100, break_long_words=False) if len(line) > 100 else [line]
        )

    lines_per_page = 35
    for page_start in range(0, len(wrapped_insights), lines_per_page):
        fig = plt.figure(figsize=(9.5, 11))
        plt.axis("off")
        page_lines = wrapped_insights[page_start:page_start + lines_per_page]
        for i, line in enumerate(page_lines):
            y = 1 - (i + 1) * 0.028
            fig.text(0.06, y, line, fontsize=10, ha="left", va="top", family="DejaVu Sans")
        pdf.savefig(fig)
        plt.close(fig)

    # === Top 5 Charts by Product ===
    products = {
        "Lutetium  (177Lu) chloride N.C.A.": "Top 5 N.C.A. Customers",
        "Lutetium (177Lu) chloride C.A": "Top 5 C.A. Customers",
        "Terbium-161 chloride n.c.a": "Top 5 Terbium Customers"
    }

    for product_name, title in products.items():
        product_df = recent_df[recent_df["Product"] == product_name]
        if product_df.empty:
            continue
        top_customers = product_df.groupby("Customer")["Total_mCi"].sum().sort_values(ascending=False).head(5).index
        plot_df = product_df[product_df["Customer"].isin(top_customers)].copy()
        plot_df["WrappedCustomer"] = plot_df["Customer"].apply(lambda x: '\n'.join(textwrap.wrap(x, 12)))
        plot_df["WeekLabel"] = plot_df["Year"].astype(str) + "-W" + plot_df["Week"].astype(str).str.zfill(2)

        pivot_df = plot_df.pivot_table(
            index="WeekLabel",
            columns="WrappedCustomer",
            values="Total_mCi",
            aggfunc="sum"
        ).fillna(0)
        pivot_df = pivot_df.reindex(sorted(
            pivot_df.index,
            key=lambda x: (int(x.split("-W")[0]), int(x.split("-W")[1]))
        ))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot_df.plot(ax=ax, marker='o')

        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel("Week of Supply", fontsize=11)
        ax.set_ylabel("Total mCi Ordered", fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Customer", bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.tight_layout(pad=2.0)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("DEBUG: PDF file size right after creation:", os.path.getsize(latest_pdf))

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
upload_to_drive(summary_pdf, f"Feedback_Report_Week_{week_num}_{year}.pdf", folder_id)
upload_to_drive(latest_copy_path, "Latest_Feedback_Report.pdf", folder_id)
upload_to_drive(week_info_path, f"Feedback_Week_Info_{week_num}_{year}.txt", folder_id)

