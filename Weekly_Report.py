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

# === Define report date values (simulate 11 days ahead) ===
today = datetime.today() + timedelta(days=5)
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
        {"role": "system", "content": "You are a senior business analyst. Based on the weekly report, provide 3 data-driven insights and 1 clear recommendation. Include the responsible account manager for each customer mentioned so the sales team can follow up effectively."},
        {"role": "user", "content": report_text}
    ]
)
insights = response.choices[0].message.content

# === Save new insight to history ===
with open(insight_history_path, "a") as f:
    f.write(f"\n\n===== Week {week_num}, {year} =====\n")
    f.write(insights)

# === PDF Generation and email sending follow ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}.pdf")
with PdfPages(latest_pdf) as pdf:
    # --- Cover Page ---
    fig = plt.figure(figsize=(9.5, 11))
    plt.axis("off")

    # Centered report title
    fig.text(0.5, 0.78, f"Weekly Orders Report – Week {week_num}, {year}",
             fontsize=26, ha="center", va="center", weight='bold')

    # Load and center the logo below the title
    logo_path = os.path.join(script_dir, "Isotopia.jpg")
    logo = mpimg.imread(logo_path)

    logo_width = 0.25  # 25% of the figure width
    logo_height = 0.12  # 12% of the figure height
    logo_x = (1 - logo_width) / 2  # centered horizontally
    logo_y = 0.60  # vertical placement under title

    ax_logo = fig.add_axes([logo_x, logo_y, logo_width, logo_height])
    ax_logo.imshow(logo)
    ax_logo.axis("off")

    # Save the cover page
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # --- STOPPED ORDERING TABLE ---
if stopped:
    stopped_df = pd.DataFrame([
        [wrap_text(name), wrap_text(mgr)]
        for name, mgr in stopped
    ], columns=["Customer", "Account Manager"])

    fig_height = max(4.5, 0.4 + 0.3 * len(stopped_df))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=stopped_df.values,
        colLabels=stopped_df.columns,
        loc="upper left",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 2.0)

    # Manually widen the "Customer" column (column index 0)
    for key, cell in table.get_celld().items():
        row, col = key
        if col == 0:
            cell.set_width(0.4)
        else:
            cell.set_width(0.15)

    ax.set_title("STOPPED ORDERING", fontsize=12, weight="bold", pad=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


    # --- DECREASED ORDERS TABLE ---
   if decreased:
    decreased_df = pd.DataFrame([
        [
            wrap_text(name),
            f"{curr - prev:+.0f}",
            f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%",
            wrap_text(mgr)
        ]
        for name, prev, curr, mgr in decreased
    ], columns=["Customer", "Change (mCi)", "% Change", "Account Manager"])

    fig_height = max(4.5, 0.4 + 0.3 * len(decreased_df))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=decreased_df.values,
        colLabels=decreased_df.columns,
        loc="upper left",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 2.0)

    for key, cell in table.get_celld().items():
        row, col = key
        if col == 0:
            cell.set_width(0.4)
        else:
            cell.set_width(0.15)

    ax.set_title("DECREASED ORDERS", fontsize=12, weight="bold", pad=30, y=1.05)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # --- INCREASED ORDERS TABLE ---
if increased:
    increased_df = pd.DataFrame([
        [
            wrap_text(name),
            f"{curr - prev:+.0f}",
            f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%",
            wrap_text(mgr)
        ]
        for name, prev, curr, mgr in increased
    ], columns=["Customer", "Change (mCi)", "% Change", "Account Manager"])

    fig_height = max(4.5, 0.4 + 0.3 * len(increased_df))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=increased_df.values,
        colLabels=increased_df.columns,
        loc="upper left",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 2.0)

    for key, cell in table.get_celld().items():
        row, col = key
        if col == 0:
            cell.set_width(0.4)
        else:
            cell.set_width(0.15)

    ax.set_title("INCREASED ORDERS", fontsize=12, weight="bold", pad=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # --- INACTIVE IN PAST 4 WEEKS TABLE ---
if inactive_recent_4:
    inactive_df = pd.DataFrame([
        [wrap_text(name)] for name in inactive_recent_4
    ], columns=["Customer"])

    fig_height = max(4.5, 0.4 + 0.3 * len(inactive_df))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=inactive_df.values,
        colLabels=inactive_df.columns,
        loc="upper left",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 2.0)

    for key, cell in table.get_celld().items():
        row, col = key
        if col == 0:
            cell.set_width(0.4)
        else:
            cell.set_width(0.15)

    ax.set_title("INACTIVE IN PAST 4 WEEKS", fontsize=12, weight="bold", pad=10)
    pdf.savefig(fig, bbox_inches="tight")
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

        pivot_df = plot_df.pivot_table(index="WeekLabel", columns="WrappedCustomer", values="Total_mCi", aggfunc="sum").fillna(0)
        pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=lambda x: (int(x.split("-W")[0]), int(x.split("-W")[1]))))
        fig, ax = plt.subplots(figsize=(8, 4.5))  # Increased size for better clarity
        pivot_df.plot(ax=ax, marker='o')

        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel("Week of Supply", fontsize=11)
        ax.set_ylabel("Total mCi Ordered", fontsize=11)
        ax.tick_params(axis='x', rotation=45)  # Rotate week labels to prevent overlap
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Customer", bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.tight_layout(pad=2.0)  # Ensure nothing is cut off or overlapping
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

# === Send Email ===
send_email(
    subject=f"Weekly Orders Report – Week {week_num}, {year}",
    body=f"""Dear team,

Attached is the weekly orders report for Week {week_num}, {year}. The report outlines key changes in customer activity, including:

Customers who increased or decreased their orders

Customers who stopped ordering entirely

Customers who were inactive in the past 4 weeks but had ordered in the 4 weeks prior

The top 5 customers for each product category: NCA, CA, and Terbium

Please let me know if you have any questions or if you'd like to add someone to the mailing list.

Best regards,
Dan
""",
    to_emails=[
    "danamit@isotopia-global.com"
],
    attachment_path=latest_pdf
)
print("✅ Report generated and emailed via Resend API.")
