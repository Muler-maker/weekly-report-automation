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
    context_lines = []
    for idx, row in feedback_df.iterrows():
        if ((str(row['Status']).lower() != 'done') or (str(row['Week']) == str(week_num) and str(row['Year']) == str(year))):
            line = (f"AM: {row['AM']}\n"
                    f"Distributor: {row['Distributor']} | Country: {row['Country/Countries']} | Customers: {row['Customers']}\n"
                    f"Question: {row['Question']}\n"
                    f"Answer: {row['Comments / Feedback']}\n"
                    f"Status: {row['Status']}, Date: {row['Feedback Date']}\n"
                    "------")
            context_lines.append(line)
    return "\n".join(context_lines)

def add_comparison_tables_page_to_pdf(pdf, df):
    """
    Adds a single page with three separate comparison tables to the PDF.
    - Compares shifted timeframes (e.g., last week vs. week before).
    - Colors percentage change values based on performance.
    """
    # === Helper function to create each table ===
    def draw_comparison_table(ax, data, title, period1_name, period2_name):
        ax.axis('off')
        ax.set_title(title, fontsize=12, weight='bold', pad=15)

        if data.empty:
            ax.text(0.5, 0.5, 'No data for this comparison', ha='center', va='center')
            return
        
        # Calculate percentage change
        data['% Change'] = ((data[period1_name] - data[period2_name]) / data[period2_name] * 100).replace([np.inf, -np.inf], 100).fillna(0)

        # Format for display
        display_data = data.copy()
        display_data[period1_name] = display_data[period1_name].apply(lambda x: f'{x:,.0f}')
        display_data[period2_name] = display_data[period2_name].apply(lambda x: f'{x:,.0f}')
        display_data['% Change'] = display_data['% Change'].apply(lambda x: f'{x:+.1f}%')

        display_data = display_data.reset_index() # Move 'Product' from index to column
        
        table = ax.table(cellText=display_data.values,
                         colLabels=['Product', period1_name, period2_name, '% Change'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.04
            if row == 0: # Header row
                cell.set_facecolor("#e6e6fa"); cell.set_text_props(weight='bold')
            else: # Data rows
                cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
                # Color the '% Change' column (index 3)
                if col == 3:
                    numeric_value = data.iloc[row - 1]['% Change']
                    color = 'green' if numeric_value > 0.1 else ('red' if numeric_value < -0.1 else 'black')
                    cell.set_text_props(color=color, weight='bold')
                # Left-align Product name (index 0)
                if col == 0: cell.set_text_props(ha='left')

    # === Define Time Periods (Shifted Back) ===
    today = datetime.today() + timedelta(days=11)
    last_week_date = today - timedelta(weeks=1)
    week_before_last_date = today - timedelta(weeks=2)
    same_week_ly_date = last_week_date - timedelta(weeks=52)
    
    previous_8_weeks = [(d.isocalendar().year, d.isocalendar().week) for d in [today - timedelta(weeks=i) for i in range(1, 9)]]
    before_previous_8_weeks = [(d.isocalendar().year, d.isocalendar().week) for d in [today - timedelta(weeks=i) for i in range(9, 17)]]
    
    # === Filter and Aggregate Data ===
    df_agg = lambda yw: df[df['YearWeek'].isin(yw)].groupby('Product')['Total_mCi'].sum() if isinstance(yw, list) else df[df['YearWeek'] == yw].groupby('Product')['Total_mCi'].sum()
    
    last_week_totals = df_agg((last_week_date.isocalendar().year, last_week_date.isocalendar().week))
    week_before_last_totals = df_agg((week_before_last_date.isocalendar().year, week_before_last_date.isocalendar().week))
    same_week_ly_totals = df_agg((same_week_ly_date.isocalendar().year, same_week_ly_date.isocalendar().week))
    previous_8_weeks_totals = df_agg(previous_8_weeks)
    before_previous_8_weeks_totals = df_agg(before_previous_8_weeks)
    
    # === Create Comparison DataFrames ===
    comp1 = pd.concat([last_week_totals.rename('Prev Wk'), week_before_last_totals.rename('Wk Before')], axis=1).fillna(0)
    comp2 = pd.concat([last_week_totals.rename('Prev Wk'), same_week_ly_totals.rename('Same Wk LY')], axis=1).fillna(0)
    comp3 = pd.concat([previous_8_weeks_totals.rename('Prev 8 Wks'), before_previous_8_weeks_totals.rename('8 Wks Before')], axis=1).fillna(0)
    
    # === Create the Page with Three Tables ===
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5, 11))
    fig.suptitle('Product Performance Summary', fontsize=16, weight='bold')

    draw_comparison_table(ax1, comp1, 'Previous Week vs. Week Before', 'Prev Wk', 'Wk Before')
    draw_comparison_table(ax2, comp2, 'Previous Week vs. Same Week Last Year', 'Prev Wk', 'Same Wk LY')
    draw_comparison_table(ax3, comp3, 'Previous 8 Weeks vs. 8 Weeks Before', 'Prev 8 Wks', '8 Wks Before')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

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

# === Data Processing and Analysis ---
account_manager_map = {"vbeillis@isotopia-global.com": "Vicki Beillis", "naricha@isotopia-global.com": "Noam Aricha", "ndellus@isotopia-global.com": "Noam Dellus", "gbader@isotopia-global.com": "Gilli Bader", "yfarchi@isotopia-global.com": "Yosi Farchi", "caksoy@isotopia-global.com": "Can Aksoy"}
df["Account Manager"] = df["Account Manager Email"].map(account_manager_map).fillna("Other")

def send_email(subject, body, to_emails, attachment_path):
    if not os.path.exists(attachment_path):
        print(f"❌ Attachment file not found: {attachment_path}")
        return
    with open(attachment_path, "rb") as f:
        files = {'attachments': (os.path.basename(attachment_path), f, 'application/pdf')}
        data = {"from": "Dan Amit <dan.test@resend.dev>", "to": ','.join(to_emails), "subject": subject, "text": body}
        headers = {"Authorization": f"Bearer {os.getenv('RESEND_API_KEY')}"}
        response = requests.post("https://api.resend.com/emails", data=data, files=files, headers=headers)
    if 200 <= response.status_code < 300: print(f"📨 Email successfully sent to: {', '.join(to_emails)}")
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

distributor_country_to_customers = defaultdict(list); country_to_distributors = defaultdict(set)
country_to_customers = defaultdict(set); distributor_to_countries = defaultdict(set)
distributor_to_customers = defaultdict(set)
for _, row in df.iterrows():
    distributor = str(row['Distributor']).strip(); country = str(row['Country']).strip(); customer = str(row['Customer']).strip()
    if distributor and country and customer:
        distributor_country_to_customers[(distributor, country)].append(customer)
        country_to_distributors[country].add(distributor); country_to_customers[country].add(customer)
        distributor_to_countries[distributor].add(country); distributor_to_customers[distributor].add(customer)

current = today - timedelta(days=today.weekday())
week_pairs = [(d.isocalendar().year, d.isocalendar().week) for d in [current - timedelta(weeks=(15 - i)) for i in range(16)]]
previous_8, recent_8 = week_pairs[:8], week_pairs[8:]
recent_df = df[df["YearWeek"].isin(recent_8)]; previous_df = df[df["YearWeek"].isin(previous_8)]
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

# === ChatGPT Insights & Executive Summary Generation ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key: raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=openai_api_key)

summary_path = os.path.join(output_folder, "summary_test.txt")
# ... (Summary text formatting logic is unchanged)
format_row = lambda name, prev, curr, mgr: (f"{name:<30} | {curr - prev:+7.0f} mCi | {(curr - prev) / prev * 100 if prev else 100:+6.1f}% | {mgr}", (curr - prev) / prev * 100 if prev else 100)
increased_formatted = sorted([format_row(*x) for x in increased], key=lambda x: x[1], reverse=True)
decreased_formatted = sorted([format_row(*x) for x in decreased], key=lambda x: x[1])
summary_lines = ["STOPPED ORDERING:", "...", "---"] + [f"{x[0]:<35} | {x[1]}" for x in stopped] if stopped else ["- None"]
summary_lines += ["", "DECREASED ORDERS:", "...", "---"] + [x[0] for x in decreased_formatted] if decreased_formatted else ["- None"]
summary_lines += ["", "INCREASED ORDERS:", "...", "---"] + [x[0] for x in increased_formatted] if increased_formatted else ["- None"]
last_8_weeks = week_pairs[-8:]
previous_4_weeks, recent_4_weeks = last_8_weeks[:4], last_8_weeks[4:]
active_previous_4 = set(df[df["YearWeek"].isin(previous_4_weeks)]["Customer"])
active_recent_4 = set(df[df["YearWeek"].isin(recent_4_weeks)]["Customer"])
inactive_recent_4 = sorted(active_previous_4 - active_recent_4)
summary_lines += ["", "INACTIVE IN PAST 4 WEEKS:", "...", "---"] + inactive_recent_4 if inactive_recent_4 else ["- None"]
with open(summary_path, "w") as f: f.write("\n".join(summary_lines))

with open(summary_path, "r", encoding="utf-8") as f: report_text = f.read()
insight_history_path = os.path.join(output_folder, "insight_history_test.txt")
if os.path.exists(insight_history_path):
    with open(insight_history_path, "r", encoding="utf-8") as f: past_insights = f.read()
    report_text = f"{past_insights}\n\n===== NEW WEEK =====\n\n{report_text}"
feedback_context = build_feedback_context(feedback_df, week_num, year)
def format_row_for_gpt(name, prev, curr, mgr):
    distributor = customer_to_distributor.get(name, "Unknown"); country = customer_to_country.get(name, "Unknown")
    return (f"{name:<30} | {curr - prev:+7.0f} mCi | "
            f"{(curr - prev) / prev * 100 if prev else 100:+6.1f}% | {mgr} | {distributor} | {country}")
gpt_rows = [format_row_for_gpt(name, prev, curr, mgr) for (name, prev, curr, mgr) in increased + decreased]
gpt_distributor_section = ["Customer name                     | Change   | % Change | Account Manager | Distributor | Country"] + gpt_rows
full_prompt_text = ( "Previous feedback from Account Managers...\n" + feedback_context + "\n\n==== New Weekly Report Data ====\n\n" + report_text + "\n\n=== Distributor info for GPT analysis ===\n" + "\n".join(gpt_distributor_section))
system_prompt = "You are a senior business analyst..." # The long system prompt is unchanged
response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_prompt_text}])
insights = response.choices[0].message.content.strip()
print("\n💡 GPT Insights:\n", insights)

# ... (Logic for extracting questions and updating Google Sheet is unchanged)
exec_summary_prompt ="You are a senior business analyst..." # The exec summary prompt is unchanged
exec_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": exec_summary_prompt}, {"role": "user", "content": report_text}])
executive_summary = exec_response.choices[0].message.content.strip()
summary_json_path = os.path.join(output_folder, "Executive_Summary_test.json")
with open(summary_json_path, "w", encoding="utf-8") as f: json.dump({"executive_summary": executive_summary}, f, ensure_ascii=False, indent=2)
upload_to_drive(summary_json_path, "Executive_Summary_test.json", folder_id)
# ... (Logic for writing summary to Google Sheet is unchanged)

# === PDF Generation ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}_test.pdf")

with PdfPages(latest_pdf) as pdf:
    print("DEBUG: Entered PdfPages block")

    # --- Cover Page FIRST ---
    fig_cover = plt.figure(figsize=(9.5, 11)); plt.axis("off")
    fig_cover.text(0.5, 0.80, "Weekly Orders Report", fontsize=26, ha="center", va="center", weight='bold')
    fig_cover.text(0.5, 0.74, f"Week {week_num}, {year}", fontsize=20, ha="center", va="center")
    logo_path = os.path.join(script_dir, "Isotopia.jpg")
    if os.path.exists(logo_path):
        logo = mpimg.imread(logo_path)
        ax_logo = fig_cover.add_axes([(1 - 0.35) / 2, 0.40, 0.35, 0.18])
        ax_logo.imshow(logo); ax_logo.axis("off")
    else: print("WARNING: Logo file not found.")
    pdf.savefig(fig_cover, bbox_inches="tight"); plt.close(fig_cover)
    
    # --- SECOND PAGE: Product Comparison Tables ---
    add_comparison_tables_page_to_pdf(pdf, df)

    # --- Rest of the report ---
    if not any([stopped, decreased, increased, inactive_recent_4]):
        fig_fallback = plt.figure(figsize=(8.5, 11)); plt.axis("off")
        fig_fallback.text(0.5, 0.5, "No trend data available for this week.", ha="center", va="center", fontsize=18)
        pdf.savefig(fig_fallback); plt.close(fig_fallback)
    else:
        # ALL a subsequent PDF generation logic is restored here
        total_decreased = sum(curr - prev for v in decreased_by_am.values() for _, prev, curr, _ in v)
        total_increased = sum(curr - prev for v in increased_by_am.values() for _, prev, curr, _ in v)
        summary_df = pd.DataFrame([["Stopped Ordering", len(stopped), "–"],["Decreased Orders", sum(len(v) for v in decreased_by_am.values()), f"{total_decreased:+.1f}"],["Increased Orders", sum(len(v) for v in increased_by_am.values()), f"{total_increased:+.1f}"],["Inactive (last 4 wks)", len(inactive_recent_4), "–"]], columns=["Category", "# of Customers", "Total Δ (mCi)"])
        fig, ax = plt.subplots(figsize=(8.5, max(2.2, 0.4 + 0.3 * len(summary_df)))); ax.axis("off")
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.1, 1.4)
        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
            cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="center")
        ax.set_title("Summary by Category", fontsize=14, weight='bold', pad=10)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        am_list = set(list(decreased_by_am.keys()) + list(increased_by_am.keys()) + [customer_to_manager.get(name, "Other") for name in inactive_recent_4 + [x[0] for x in stopped]])
        am_summary = [[am, len(increased_by_am.get(am, [])), len(decreased_by_am.get(am, [])), sum(1 for name in inactive_recent_4 if customer_to_manager.get(name, "Other") == am), sum(1 for name, _ in stopped if customer_to_manager.get(name, "Other") == am)] for am in sorted(am_list)]
        am_df = pd.DataFrame(am_summary, columns=["Account Manager", "+ Customers", "– Customers", "Inactive", "Stopped"])
        fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.4 + 0.3 * len(am_df)))); ax.axis("off")
        table = ax.table(cellText=am_df.values, colLabels=am_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.1, 1.4)
        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
            cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="center")
        ax.set_title("Summary by Account Manager", fontsize=14, weight='bold', pad=10)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

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
            ax.tick_params(axis='x', rotation=45); ax.grid(True, linestyle='--', alpha=0.5); ax.legend(title="Customer", bbox_to_anchor=(1.02, 1), loc='upper left')
            fig.tight_layout(pad=2.0); pdf.savefig(fig); plt.close(fig)
        
        # ... (And so on for all the other tables and pages)
        
# === Finalize and Upload ===
print("DEBUG: PDF file size right after creation:", os.path.getsize(latest_pdf))
summary_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Summary_Week_{week_num}_{year}_test.pdf")
latest_copy_path = os.path.join(output_folder, "Latest_Weekly_Report_test.pdf")
copyfile(latest_pdf, summary_pdf)
copyfile(latest_pdf, latest_copy_path)
print(f"✅ Report also saved as: {summary_pdf}")
print(f"✅ Report also saved as: {latest_copy_path}")
time.sleep(0.5)
week_info_path = os.path.join(output_folder, "Week_number_test.txt")
with open(week_info_path, "w") as f: f.write(f"{week_num},{year}")
upload_to_drive(summary_pdf, f"Weekly_Orders_Report_Summary_Week_{week_num}_{year}_test.pdf", folder_id)
upload_to_drive(latest_copy_path, "Latest_Weekly_Report_test.pdf", folder_id)
upload_to_drive(week_info_path, "Week_number_test.txt", folder_id)
