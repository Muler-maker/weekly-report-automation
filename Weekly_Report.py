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
from email.message import EmailMessage
from openai import OpenAI
import requests
import base64

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

# === Define email sending function using Resend API ===
def send_email(subject, body, to_emails, attachment_path):
    with open(attachment_path, "rb") as f:
        encoded_file = base64.b64encode(f.read()).decode()

    data = {
        "from": "Dan Amit <onboarding@resend.dev>",
        "to": to_emails,
        "subject": subject,
        "text": body,
        "attachments": [
            {
                "filename": os.path.basename(attachment_path),
                "content": encoded_file,
                "contentType": "application/pdf"
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('RESEND_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.resend.com/emails", json=data, headers=headers)

    if response.status_code >= 200 and response.status_code < 300:
        print(f"\U0001F4E7 Email sent to: {', '.join(to_emails)} via Resend API")
    else:
        print("\u274C Failed to send email:", response.status_code, response.text)

# === Define report date values ===
today = datetime.today()
week_num = today.isocalendar().week
year = today.year

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
] + [x[0] for x in stopped] if stopped else ["- None"]

summary_lines += [
    "",
    "DECREASED ORDERS:",
    "These customers ordered less in the last 8 weeks compared to the 8 weeks prior.",
    "Customer Name                   | Change   | % Change",
    "-----------------------------------------------------"
] + [x[0] for x in decreased_formatted] if decreased_formatted else ["- None"]

summary_lines += [
    "",
    "INCREASED ORDERS:",
    "These customers increased their order amounts in the last 8 weeks compared to the 8 weeks prior.",
    "Customer Name                   | Change   | % Change",
    "-----------------------------------------------------"
] + [x[0] for x in increased_formatted] if increased_formatted else ["- None"]

summary_lines += [
    "",
    "INACTIVE IN PAST 4 WEEKS:",
    "Customers who ordered in the previous 4 weeks but not in the most recent 4.",
    "Customer Name",
    "-------------------------------"
] + inactive_recent_4 if inactive_recent_4 else ["- None"]

# Save summary
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
        {"role": "system", "content": "You are a senior business analyst. Based on the weekly report, provide 3 data-driven insights and 1 clear recommendation, each highlighting specific customers, trends, or changes in behavior that should be acted upon. Prioritize actionable items that could guide follow-up by the commercial or support teams."},
        {"role": "user", "content": report_text}
    ]
)
insights = response.choices[0].message.content

# Save new insight to history
with open(insight_history_path, "a") as f:
    f.write(f"\n\n===== Week {week_num}, {year} =====\n")
    f.write(insights)

# === PDF Generation ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}.pdf")
with PdfPages(latest_pdf) as pdf:
    fig = plt.figure(figsize=(9.5, 11))
    plt.axis("off")
    fig.text(0.5, 0.5, f"Weekly Orders Report – Week {week_num}, {year}", fontsize=26, ha="center", va="center", weight='bold')
    pdf.savefig(fig)
    plt.close(fig)

    wrapped_lines = []
    for line in summary_lines:
        wrapped_lines.extend(textwrap.wrap(line, width=100, break_long_words=False) if len(line) > 100 else [line])
    for p in range(0, len(wrapped_lines), 40):
        fig = plt.figure(figsize=(9.5, 11))
        plt.axis("off")
        for i, line in enumerate(wrapped_lines[p:p + 40]):
            y = 1 - (i + 1) * 0.028
            style = {"fontsize": 10, "ha": "left", "va": "top", "family": "DejaVu Sans"}
            if line.isupper() and ":" in line:
                style.update({"fontsize": 12, "weight": "bold"})
            fig.text(0.06, y, line, **style)
        pdf.savefig(fig)
        plt.close(fig)

    # === GPT Insight Page ===
    insight_lines = insights.split("\n")

    wrapped_insights = []
    for line in insight_lines:
        wrapped_insights.extend(textwrap.wrap(line, width=100, break_long_words=False) if len(line) > 100 else [line])
    fig = plt.figure(figsize=(9.5, 11))
    plt.axis("off")
    for i, line in enumerate(wrapped_insights):
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
        plot_df["WeekLabel"] = plot_df["Year"].astype(str) + "-W" + plot_df["Week"].astype(str).str.zfill(2)
        pivot_df = plot_df.pivot_table(index="WeekLabel", columns="Customer", values="Total_mCi", aggfunc="sum").fillna(0)
        pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=lambda x: (int(x.split("-W")[0]), int(x.split("-W")[1]))))
        fig, ax = plt.subplots(figsize=(9.5, 6))
        pivot_df.plot(ax=ax, marker='o')
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel("Production Week", fontsize=10)
        ax.set_ylabel("Total mCi Ordered", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Customer", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# === Send Email ===
send_email(
    subject=f"Weekly Orders Report – Week {week_num}, {year}",
    body=f"""Dear team,

Attached is the weekly orders report for week {week_num}, {year}, including key trends and insights on customer ordering behavior.

Best regards,
Dan
""",
    to_emails=[
        "danamit@isotopia-global.com"
    ],
    attachment_path=latest_pdf
)
print("✅ Report generated and emailed via Resend API.")
