import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pandas as pd
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
def wrap_text(text, width=20):
    return '\n'.join(textwrap.wrap(str(text), width=width))

# === Set global font style ===
plt.rcParams["font.family"] = "DejaVu Sans"

# === Set local output folder ===
output_folder = os.path.join(os.getcwd(), 'Reports')
os.makedirs(output_folder, exist_ok=True)

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
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_drive(file_path, file_name, folder_id=None):
    drive_service = build('drive', 'v3', credentials=creds)

    # Search for an existing file with the same name in the target folder
    query = f"name = '{file_name}'"
    if folder_id:
        query += f" and '{folder_id}' in parents"

    existing_files = drive_service.files().list(
        q=query,
        fields="files(id, name)"
    ).execute().get('files', [])

    if existing_files:
        # File exists — overwrite it using update
        file_id = existing_files[0]['id']
        media = MediaFileUpload(file_path, mimetype='application/pdf', resumable=True)
        updated_file = drive_service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        print(f"🔁 Overwritten on Google Drive: {file_name} (ID: {file_id})")
    else:
        # File doesn't exist — create a new one
        file_metadata = {'name': file_name}
        if folder_id:
            file_metadata['parents'] = [folder_id]

        media = MediaFileUpload(file_path, mimetype='application/pdf')
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"✅ Uploaded to Google Drive: {file_name} (ID: {uploaded_file.get('id')})")

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
    "Catalogue description (sold as)": "Product"
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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
with open(summary_path, "r") as f:
    report_text = f.read()

insight_history_path = os.path.join(output_folder, "insight_history.txt")
if os.path.exists(insight_history_path):
    with open(insight_history_path, "r") as f:
        past_insights = f.read()
    report_text = past_insights + "\n\n" + report_text

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """You are a senior business analyst. Based on the weekly report, provide clear and concise action items organized by Account Manager. Each action item should identify the relevant customer, the issue or opportunity, and a suggested next step.

Use the following considerations during your analysis:
1. COMISSÃO NACIONAL DE ENERGIA NUCLEAR (CNEN) is expected to place orders every two weeks on even-numbered weeks. Flag any deviation from this pattern.
2. Treat the following customers as a single group under 'Sinotau' when evaluating order volume trends:
   - Glotope (Mianyang) Advanced Pharmaceutical Technology Ltd
   - Jiangsu Sinotau Molecular Imaging Science & Technology Co. LTD
   - Sinotau Pharmaceutical Group (Beijing)
   - Sinotau Pharmaceutical Group (Guangdong)
   - Sinotau Pharmaceutical Group (Sichuan)
3. Treat the following customers as a single group under 'Seibersdorf Laboratories':
   - Seibersdorf Laboratories
   - Seibersdorf Laboratories (Blue Earth Therapeutics)
   - Seibersdorf Laboratories (Debiopharm)
   - Seibersdorf Laboratories (Philochem)
   - Seibersdorf Laboratories (Starget Pharma)
4. Only include Sinotau or Seibersdorf if there’s a noteworthy insight.
5. Focus on abnormal behavior, order spikes/drops, or lack of recent activity. Avoid stating the obvious.
6. Keep action items short, specific, and helpful."""
        },
        {
            "role": "user",
            "content": report_text
        }
    ]
)
insights = response.choices[0].message.content
print("\n💡 GPT Insights:\n", insights)

# === Save new insight to history ===
with open(insight_history_path, "a") as f:
    f.write(f"\n\n===== Week {week_num}, {year} =====\n")
    f.write(insights)

# === TEST PDF Generation ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}.pdf")

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

with PdfPages(latest_pdf) as pdf:
    # Page 1: Cover
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size
    plt.axis("off")
    fig.text(0.5, 0.8, "Weekly Orders Report – TEST COVER", fontsize=20, ha="center")
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: GPT Insights mock
    lines = [
        "**Noam Aricha:**",
        "- Follow up with CNEN about skipped week 22",
        "**Gilli Bader:**",
        "- Customer X dropped 60% vs. prior period",
    ]

    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis("off")
    for i, line in enumerate(lines):
        y = 1 - (i + 1) * 0.05
        if line.startswith("**") and line.endswith("**"):
            text = line.strip("*")
            weight = "bold"
        else:
            text = line
            weight = "normal"
        fig.text(0.06, y, text, fontsize=10, ha="left", va="top", weight=weight)
    pdf.savefig(fig)
    plt.close(fig)

print(f"✅ TEST PDF saved: {latest_pdf}")
