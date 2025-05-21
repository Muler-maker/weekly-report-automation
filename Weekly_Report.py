# Adjusted script for running in local Jupyter Notebook and Task Scheduler

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

# === Define email sending function ===
def send_email(subject, body, to_emails, attachment_path):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "isotopiadan@gmail.com"
    app_password = "nrsrroyqenqtgzgj"  # <-- Replace with your valid Gmail App Password

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)
        msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)

    with smtplib.SMTP(smtp_server, smtp_port) as smtp:
        smtp.starttls()
        smtp.login(sender_email, app_password)
        smtp.send_message(msg)

    print(f"📧 Email sent to: {', '.join(to_emails)}")

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

# Generate graphs
product_graphs = {
    "Lutetium  (177Lu) chloride N.C.A.": "Top Lutetium-177 N.C.A Customers",
    "Lutetium (177Lu) chloride C.A": "Top Lutetium-177 C.A Customers",
    "Terbium-161 chloride n.c.a": "Top Terbium-161 Customers"
}
chart_paths = []
def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name)

for product, title in product_graphs.items():
    filtered = recent_df[recent_df["Product"] == product]
    grouped = filtered.groupby(["Customer", "Week"]).agg({"Total_mCi": "sum"}).reset_index()
    top_customers = grouped.groupby("Customer")["Total_mCi"].sum().nlargest(5).index
    data = grouped[grouped["Customer"].isin(top_customers)]
    if not data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x="Week", y="Total_mCi", hue="Customer", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Week Number")
        ax.set_ylabel("Total Ordered (mCi)")
        ax.grid(True)
        ax.legend(title="Customer", loc="upper left", bbox_to_anchor=(1.02, 1))
        safe_filename = sanitize_filename(title) + ".png"
        chart_path = os.path.join(output_folder, safe_filename)
        fig.savefig(chart_path, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append((title, chart_path))

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

# Save PDF
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

    # Chart pages
    

    for title, path in chart_paths:
        img = plt.imread(path)
        fig, ax = plt.subplots(figsize=(9.5, 11))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)

    



# === Send the email ===
send_email(
    subject=f"Weekly Orders Report – Week {week_num}, {year}",
    body=f"""Dear team,

Attached is the weekly orders report for week {week_num}, {year}, including key trends and insights on customer ordering behavior.

Please review at your convenience. Let me know if you have any questions or would like to discuss the data in more detail.

Best regards,
Dan""",
    to_emails=["danamit@isotopia-global.com"],
    attachment_path=latest_pdf
)
