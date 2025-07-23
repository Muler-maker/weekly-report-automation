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

# === Utility Functions ===
def wrap_text(text, width=20):
    return '\n'.join(textwrap.wrap(str(text), width=width))


def build_feedback_context(feedback_df, week_num, year):
    """
    Formats recent feedback for use in the prompt.
    Only includes items not marked 'done', or items from the current week/year.
    """
    context_lines = []
    for idx, row in feedback_df.iterrows():
        if (
            (str(row['Status']).lower() != 'done')
            or (str(row['Week']) == str(week_num) and str(row['Year']) == str(year))
        ):
            line = (
                f"Account Manager: {row['AM']}\n"
                f"Distributor: {row['Distributor']} | Country: {row['Country/Countries']} | Customers: {row['Customers']}\n"
                f"Last question: {row['Question']}\n"
                f"AM answer: {row['Comments / Feedback']}\n"
                f"Status: {row['Status']}, Date: {row['Feedback Date']}\n"
                "------"
            )
            context_lines.append(line)
    return "\n".join(context_lines)

# --- Your utility/setup variables here ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1uJAArBkHXgFvY_JPIieKRjiOYd9-Ys7Ii9nXT3fWbUg"

import re
from datetime import datetime

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


df["Manager"] = df["Sales Manager (from Company name)"]

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
df = df[df["Region (from Company name)"] == "North America"]  # US team filter
from collections import defaultdict

# Mapping: distributor & country → customers
distributor_country_to_customers = defaultdict(list)
country_to_distributors = defaultdict(set)
country_to_customers = defaultdict(set)
distributor_to_countries = defaultdict(set)
distributor_to_customers = defaultdict(set)

for _, row in df.iterrows():
    distributor = str(row['Distributor']).strip()
    country = str(row['Country']).strip()
    customer = str(row['Customer']).strip()

    if distributor and country and customer:
        distributor_country_to_customers[(distributor, country)].append(customer)
        country_to_distributors[country].add(distributor)
        country_to_customers[country].add(customer)
        distributor_to_countries[distributor].add(country)
        distributor_to_customers[distributor].add(customer)

# === Define 16-week window ===
current = today - timedelta(days=today.weekday())
week_pairs = [(d.isocalendar().year, d.isocalendar().week) for d in [current - timedelta(weeks=(15 - i)) for i in range(16)]]
previous_8, recent_8 = week_pairs[:8], week_pairs[8:]
recent_df = df[df["YearWeek"].isin(recent_8)]
previous_df = df[df["YearWeek"].isin(previous_8)]

# === Trends Analysis ===
recent_totals = recent_df.groupby("Customer")["Total_mCi"].sum()
previous_totals = previous_df.groupby("Customer")["Total_mCi"].sum()
customer_to_manager = df.set_index("Customer")["Manager"].to_dict()

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
from collections import defaultdict

increased_by_am = defaultdict(list)
decreased_by_am = defaultdict(list)

for row in increased:
    increased_by_am[row[3]].append(row)  # row = (customer, prev, curr, manager)

for row in decreased:
    decreased_by_am[row[3]].append(row)

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
    "Customer name                     | Change   | % Change | Manager | Distributor | Country"
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
    "Customers who stopped ordering in the last 8 weeks but did order in the 8 weeks before.",
    "Customer Name                     | Manager",
    "---------------------------------------------------"
] + [f"{x[0]:<35} | {x[1]}" for x in stopped] if stopped else ["- None"]

summary_lines += [
    "",
    "DECREASED ORDERS:",
    "These customers ordered less in the last 8 weeks compared to the 8 weeks prior.",
    "Customer Name                     | Change   | % Change | Manager",
    "--------------------------------------------------------------------------"
] + [x[0] for x in decreased_formatted] if decreased_formatted else ["- None"]

summary_lines += [
    "",
    "INCREASED ORDERS:",
    "These customers increased their order amounts in the last 8 weeks compared to the 8 weeks prior.",
    "Customer Name                     | Change   | % Change | Manager",
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
system_prompt = """
You are a senior business analyst.

Analyze the weekly sales report and generate a summary of notable trends for each Sales Manager.

Structure the insights in this order:
1. Distributor-level trends (e.g., deviations from expected order patterns)
2. Country-level patterns (e.g., regional drops, spikes, or shifts)
3. Customer-specific observations (e.g., inactivity, growth, or notable behavior)

For each Sales Manager:
- Begin their section with their full name on a line by itself (e.g., John Smith).
- If no trends are observed, write: "No significant insights for this week."

Do not include any questions, follow-ups, or metadata.
Only provide concise trend summaries using plain text.
"""
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
insights = executive_summary
print("\n💡 GPT Insights:\n", insights)

import re
from datetime import datetime


# Save executive summary to JSON
summary_json_path = os.path.join(output_folder, "US_Executive_Summary.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "executive_summary": executive_summary
    }, f, ensure_ascii=False, indent=2)

# Upload JSON to Google Drive (replacing existing file if any)
upload_to_drive(summary_json_path, "US_Executive_Summary.json", folder_id)
# === Save Executive Summary also into Google Sheet ===
# Authenticate again if needed
gsheet_service = build('sheets', 'v4', credentials=creds)

# Spreadsheet ID for your target Google Sheet
summary_spreadsheet_id = "1otpE-pUer9Hhm3rw0858yh7siIG5VSTmAgvNLUCNujY"  # <-- Replace this with your Google Sheet ID

# Check if the sheet exists, if not create it
sheet_metadata = gsheet_service.spreadsheets().get(spreadsheetId=summary_spreadsheet_id).execute()
sheet_titles = [s['properties']['title'] for s in sheet_metadata['sheets']]

if "Executive Summary" not in sheet_titles:
    requests_body = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": "Executive Summary"
                    }
                }
            }
        ]
    }
    gsheet_service.spreadsheets().batchUpdate(
        spreadsheetId=summary_spreadsheet_id,
        body=requests_body
    ).execute()

# Clear existing content before writing new one
gsheet_service.spreadsheets().values().clear(
    spreadsheetId=summary_spreadsheet_id,
    range="US Executive Summary!A1"
).execute()

# Write executive summary into A1
gsheet_service.spreadsheets().values().update(
    spreadsheetId=summary_spreadsheet_id,
    range="Executive Summary!A1",
    valueInputOption="RAW",
    body={
        "values": [[executive_summary]]
    }
).execute()

print("✅ Executive Summary successfully written into Google Sheet.")

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

        fig.text(0.5, 0.80, "US Weekly Orders Report", fontsize=26, ha="center", va="center", weight='bold')
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
        # === Summary Table: Overall Categories ===
        total_decreased = sum(curr - prev for v in decreased_by_am.values() for _, prev, curr, _ in v)
        total_increased = sum(curr - prev for v in increased_by_am.values() for _, prev, curr, _ in v)

        summary_df = pd.DataFrame([
            ["Stopped Ordering", len(stopped), "–"],
            ["Decreased Orders", sum(len(v) for v in decreased_by_am.values()), f"{total_decreased:+.1f}"],
            ["Increased Orders", sum(len(v) for v in increased_by_am.values()), f"{total_increased:+.1f}"],
            ["Inactive (last 4 wks)", len(inactive_recent_4), "–"]
        ], columns=["Category", "# of Customers", "Total Δ (mCi)"])

        fig_height = max(2.2, 0.4 + 0.3 * len(summary_df))
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.1, 1.4)

        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.2
            if row == 0:
                cell.set_facecolor("#e6e6fa")
                cell.set_text_props(weight='bold', ha="center")
            else:
                cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
                cell.set_text_props(ha="center")

        ax.set_title("Summary by Category", fontsize=14, weight='bold', pad=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # === Summary Table: By Account Manager ===
        am_list = set(
            list(decreased_by_am.keys()) +
            list(increased_by_am.keys()) +
            [customer_to_manager.get(name, "Other") for name in inactive_recent_4 + [x[0] for x in stopped]]
        )
        am_summary = []
        for am in sorted(am_list):
            num_plus = len(increased_by_am.get(am, []))
            num_minus = len(decreased_by_am.get(am, []))
            num_inactive = sum(1 for name in inactive_recent_4 if customer_to_manager.get(name, "Other") == am)
            num_stopped = sum(1 for name, _ in stopped if customer_to_manager.get(name, "Other") == am)
            am_summary.append([am, num_plus, num_minus, num_inactive, num_stopped])

        am_df = pd.DataFrame(am_summary, columns=["Account Manager", "+ Customers", "– Customers", "Inactive", "Stopped"])
        fig_height = max(2.5, 0.4 + 0.3 * len(am_df))
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=am_df.values,
            colLabels=am_df.columns,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.1, 1.4)

        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.2
            if row == 0:
                cell.set_facecolor("#e6e6fa")
                cell.set_text_props(weight='bold', ha="center")
            else:
                cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
                cell.set_text_props(ha="center")

        ax.set_title("Summary by Account Manager", fontsize=14, weight='bold', pad=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
  # === Add Top 5 Charts by Product ===
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

            # Sort weeks in calendar order
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
            pdf.savefig(fig)
            plt.close(fig)

        # --- STOPPED ORDERING TABLE ---
        if stopped:
            stopped_sorted = sorted(stopped, key=lambda x: customer_to_manager.get(x[0], "Other"))
            stopped_df = pd.DataFrame([
                [name, wrap_text(customer_to_manager.get(name, "Other"))]
                for name, _ in stopped_sorted
            ], columns=["Customer", "Account Manager"])

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

            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2
                if row == 0:
                    cell.set_facecolor("#e6e6fa")
                    cell.set_text_props(weight='bold', ha="left")
                else:
                    cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
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
            for am, am_rows in sorted(decreased_by_am.items()):
                decreased_df = pd.DataFrame([
                    [
                        name,
                        f"{curr - prev:+.0f}",
                        f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%",
                        wrap_text(am),
                        (curr - prev) / prev * 100 if prev else 100
                    ]
                    for name, prev, curr, _ in am_rows
                ], columns=["Customer", "Change (mCi)", "% Change", "Account Manager", "PercentValue"])

                decreased_df = decreased_df.sort_values("PercentValue", ascending=True).drop(columns="PercentValue")
                fig_height = max(4.5, 0.4 + 0.3 * len(decreased_df))
                fig, ax = plt.subplots(figsize=(11, fig_height + 1))
                ax.axis("off")

                colWidths = [0.64, 0.11, 0.11, 0.14]
                table = ax.table(
                    cellText=decreased_df.values,
                    colLabels=decreased_df.columns,
                    loc="upper left",
                    cellLoc="center",
                    colWidths=colWidths
                )

                table.auto_set_font_size(False)
                table.set_fontsize(7.5)
                table.scale(1.0, 1.4)

                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2
                    if row == 0:
                        cell.set_facecolor("#e6e6fa")
                        cell.set_text_props(weight='bold')
                        cell._loc = 'left' if col == 0 else 'center'
                    else:
                        cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
                        cell.set_text_props(ha="left" if col == 0 else "center")
                        cell._loc = 'left' if col == 0 else 'center'

                ax.set_title(f"DECREASED ORDERS – {am}", fontsize=14, weight="bold", pad=15)
                fig.text(
                    0.5, 0.87,
                    f"Customers managed by {am} who decreased orders in the last 8 weeks vs. prior 8.",
                    fontsize=10, ha="center"
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # --- INCREASED ORDERS TABLE ---
        if increased:
            for am, am_rows in sorted(increased_by_am.items()):
                increased_df = pd.DataFrame([
                    [
                        name,
                        f"{curr - prev:+.0f}",
                        f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%",
                        wrap_text(am),
                        (curr - prev) / prev * 100 if prev else 100
                    ]
                    for name, prev, curr, _ in am_rows
                ], columns=["Customer", "Change (mCi)", "% Change", "Account Manager", "PercentValue"])

                increased_df = increased_df.sort_values("PercentValue", ascending=False).drop(columns="PercentValue")
                fig_height = max(4.5, 0.4 + 0.3 * len(increased_df))
                fig, ax = plt.subplots(figsize=(11, fig_height + 1))
                ax.axis("off")

                colWidths = [0.64, 0.11, 0.11, 0.14]
                table = ax.table(
                    cellText=increased_df.values,
                    colLabels=increased_df.columns,
                    loc="upper left",
                    cellLoc="center",
                    colWidths=colWidths
                )

                table.auto_set_font_size(False)
                table.set_fontsize(7.5)
                table.scale(1.0, 1.4)

                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2
                    if row == 0:
                        cell.set_facecolor("#e6e6fa")
                        cell.set_text_props(weight='bold')
                        cell._loc = 'left' if col == 0 else 'center'
                    else:
                        cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
                        cell.set_text_props(ha="left" if col == 0 else "center")
                        cell._loc = 'left' if col == 0 else 'center'

                ax.set_title(f"INCREASED ORDERS – {am}", fontsize=14, weight="bold", pad=15)
                fig.text(
                    0.5, 0.87,
                    f"Customers managed by {am} who increased orders in the last 8 weeks vs. prior 8.",
                    fontsize=10, ha="center"
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # --- INACTIVE IN PAST 4 WEEKS TABLE ---
        if inactive_recent_4:
            inactive_sorted = sorted(inactive_recent_4, key=lambda name: customer_to_manager.get(name, "Other"))
            inactive_df = pd.DataFrame([
                [name, wrap_text(customer_to_manager.get(name, "Other"))]
                for name in inactive_sorted
            ], columns=["Customer", "Account Manager"])

            fig_height = max(4.5, 0.4 + 0.3 * len(inactive_df))
            fig, ax = plt.subplots(figsize=(11, fig_height + 1))
            ax.axis("off")

            table = ax.table(
                cellText=inactive_df.values,
                colLabels=inactive_df.columns,
                loc="upper left",
                cellLoc="left",
                colWidths=[0.8, 0.2]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1.0, 1.4)

            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2
                if row == 0:
                    cell.set_facecolor("#e6e6fa")
                    cell.set_text_props(weight='bold', ha="left")
                else:
                    cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
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

def make_text_page(title, text, lines_per_page=35):
    import matplotlib.pyplot as plt
    import textwrap

    wrapped_text = textwrap.wrap(text, width=100, break_long_words=False)
    fig = plt.figure(figsize=(9.5, 11))
    plt.axis("off")

    fig.text(0.5, 0.96, title, fontsize=16, weight="bold", ha="center")

    for i, line in enumerate(wrapped_text[:lines_per_page]):
        y = 0.91 - i * 0.028
        fig.text(0.06, y, line, fontsize=10, ha="left", va="top")

    return fig

# Later in the script (not indented)
pdf.savefig(make_text_page("💡 GPT Insights", insights))



# === After the PDF file is closed ===
print("DEBUG: PDF file size right after creation:", os.path.getsize(latest_pdf))

# === Save report with additional filenames ===
summary_pdf = os.path.join(output_folder, f"Weekly_US_Orders_Report_Summary_Week_{week_num}_{year}.pdf")
latest_copy_path = os.path.join(output_folder, "Latest_US_Weekly_Report.pdf")
copyfile(latest_pdf, summary_pdf)
copyfile(latest_pdf, latest_copy_path)
print(f"✅ Report also saved as: {summary_pdf}")
print(f"✅ Report also saved as: {latest_copy_path}")

# === Add a short pause to ensure files are fully written and closed ===
import time
time.sleep(0.5)

# === Upload PDFs to Google Drive Folder ===
upload_to_drive(summary_pdf, f"Weekly_US_Orders_Report_Summary_Week_{week_num}_{year}.pdf", folder_id)
upload_to_drive(latest_copy_path, "Latest_US_Weekly_Report.pdf", folder_id)
