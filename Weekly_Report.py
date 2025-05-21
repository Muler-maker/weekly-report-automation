# Adjusted script with ChatGPT insights integration using OpenAI v1.x and SendGrid

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os, re
from shutil import copyfile
import textwrap
import smtplib
from email.message import EmailMessage
from openai import OpenAI

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

# === Define email sending function using SendGrid ===
def send_email(subject, body, to_emails, attachment_path):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "Dan Amit <your_verified_email@domain.com>"  # Replace with your verified SendGrid sender
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)
        msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)

    with smtplib.SMTP("smtp.sendgrid.net", 587) as smtp:
        smtp.starttls()
        smtp.login("apikey", os.getenv("SENDGRID_API_KEY"))
        smtp.send_message(msg)

    print(f"📧 Email sent to: {', '.join(to_emails)} via SendGrid")

# === Define report date values ===
today = datetime.today()
week_num = today.isocalendar().week
year = today.year

# === Prepare weekly report data ===
df = df.rename(columns={
    "The customer": "Customer",
    "Total amount ordered (mCi)": "Total_mCi",
    "Year": "Year",
    "Week number for Activity vs Projection": "Week",
    "Shipping Status": "ShippingStatus",
    "Catalogue description (sold as)": "Product"
})

# Filter relevant shipping statuses
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

# Define 16-week range (8 + 8 weeks)
current = today - timedelta(days=today.weekday())
week_pairs = [(d.isocalendar().year, d.isocalendar().week) for d in [current - timedelta(weeks=(15 - i)) for i in range(16)]]
previous_8, recent_8 = week_pairs[:8], week_pairs[8:]
recent_df = df[df["YearWeek"].isin(recent_8)]
previous_df = df[df["YearWeek"].isin(previous_8)]

# Trends
recent_totals = recent_df.groupby("Customer")["Total_mCi"].sum()
previous_totals = previous_df.groupby("Customer")["Total_mCi"].sum()
all_customers = set(df["Customer"])
increased, decreased, stopped = [], [], []
for customer in all_customers:
    prev = previous_totals.get(customer, 0)
    curr = recent_totals.get(customer, 0)
    if prev == 0 and curr > 0:
        increased.append((customer, prev, curr))
    elif curr == 0 and prev > 0:
        stopped.append((customer,))
    elif curr > prev:
        increased.append((customer, prev, curr))
    elif curr < prev:
        decreased.append((customer, prev, curr))

# Inactive in last 4 weeks (but active before)
last_8_weeks = week_pairs[-8:]
previous_4_weeks, recent_4_weeks = last_8_weeks[:4], last_8_weeks[4:]
active_previous_4 = set(df[df["YearWeek"].isin(previous_4_weeks)]["Customer"])
active_recent_4 = set(df[df["YearWeek"].isin(recent_4_weeks)]["Customer"])
inactive_recent_4 = sorted(active_previous_4 - active_recent_4)

# Format summary
format_row = lambda name, prev, curr: (f"{name:<30} | {curr - prev:+7.0f} mCi | {(curr - prev) / prev * 100 if prev else 100:+6.1f}%", (curr - prev) / prev * 100 if prev else 100)

increased_formatted = [format_row(*x) for x in increased]
decreased_formatted = [format_row(*x) for x in decreased]

increased_formatted.sort(key=lambda x: x[1], reverse=True)
decreased_formatted.sort(key=lambda x: x[1])

summary_lines = [
    "STOPPED ORDERING:",
    "Customers who stopped ordering in the last 4 weeks but did order in the 4 weeks before.",
    "Customer Name",
    "-------------------------------"
]
summary_lines += [x[0] for x in stopped] if stopped else ["- None"]

summary_lines += [
    "",
    "DECREASED ORDERS:",
    "These customers ordered less in the last 8 weeks compared to the 8 weeks prior.",
    "Customer Name                   | Change   | % Change",
    "-----------------------------------------------------"
]
summary_lines += [x[0] for x in decreased_formatted] if decreased_formatted else ["- None"]

summary_lines += [
    "",
    "INCREASED ORDERS:",
    "These customers increased their order amounts in the last 8 weeks compared to the 8 weeks prior.",
    "Customer Name                   | Change   | % Change",
    "-----------------------------------------------------"
]
summary_lines += [x[0] for x in increased_formatted] if increased_formatted else ["- None"]

summary_lines += [
    "",
    "INACTIVE IN PAST 4 WEEKS:",
    "Customers who ordered in the previous 4 weeks but not in the most recent 4.",
    "Customer Name",
    "-------------------------------"
]
summary_lines += inactive_recent_4 if inactive_recent_4 else ["- None"]

# Save summary as text for ChatGPT
summary_path = os.path.join(output_folder, "summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

# === ChatGPT integration using OpenAI v1.x ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open(summary_path, "r") as f:
    report_text = f.read()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You're an expert analyst. Provide 3 insights and 1 recommendation based on the weekly report."},
        {"role": "user", "content": report_text}
    ]
)

insights = response.choices[0].message.content
insights_path = os.path.join(output_folder, "insights.txt")
with open(insights_path, "w") as f:
    f.write(insights)

# === Generate the PDF with summary, insights, and charts ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}.pdf")
with PdfPages(latest_pdf) as pdf:
    # Cover page
    fig = plt.figure(figsize=(9.5, 11))
    plt.axis("off")
    fig.text(0.5, 0.5, f"Weekly Orders Report – Week {week_num}, {year}", fontsize=22, ha="center", va="center", weight='bold')
    pdf.savefig(fig)
    plt.close(fig)

    # Summary pages
    wrapped_lines = []
    for line in summary_lines:
        wrapped_lines.extend(textwrap.wrap(line, 90) if len(line) > 90 else [line])
    for p in range(0, len(wrapped_lines), 40):
        fig = plt.figure(figsize=(9.5, 11))
        plt.axis("off")
        for i, line in enumerate(wrapped_lines[p:p + 40]):
            y = 1 - (i + 1) * 0.025
            fig.text(0.06, y, line, fontsize=9, ha="left", va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

    # Insights page
    insight_lines = insights.split("\n")
    fig = plt.figure(figsize=(9.5, 11))
    plt.axis("off")
    for i, line in enumerate(insight_lines):
        y = 1 - (i + 1) * 0.03
        fig.text(0.06, y, line, fontsize=10, ha="left", va="top")
    pdf.savefig(fig)
    plt.close(fig)

# === Send the email ===
send_email(
    subject=f"Weekly Orders Report – Week {week_num}, {year}",
    body="Dear team,\n\nAttached is the weekly orders report including trends and insights.\n\nBest,\nDan",
    to_emails=["danamit@isotopia-global.com"],
    attachment_path=latest_pdf
)

print("✅ Report generated and emailed via SendGrid.")
