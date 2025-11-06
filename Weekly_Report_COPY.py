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
        question_clean = re.sub(metadata_pattern, "", text).strip()
        return question_clean, distributors, countries, customers

    metadata_pattern2 = r"\(\s*Distributor:\s*(.*?)\s*;\s*Country:\s*(.*?)\s*\)\s*$"
    match2 = re.search(metadata_pattern2, text)

    if match2:
        distributors = [x.strip() for x in re.split(r",\s*", match2.group(1))]
        countries = [x.strip() for x in re.split(r",\s*", match2.group(2))]
        customers_set = set()
        for distributor in distributors:
            for country in countries:
                customers_set.update(distributor_country_to_customers.get((distributor, country), []))
        customers = sorted(customers_set)
        question_clean = re.sub(metadata_pattern2, "", text).strip()
        return question_clean, distributors, countries, customers

    question_clean = re.sub(r"\(.*?\)\s*$", "", text).strip()
    return question_clean, ["N/A"], ["N/A"], ["N/A"]

def build_feedback_context(feedback_df, week_num, year):
    context_lines = []
    for idx, row in feedback_df.iterrows():
        if ((str(row['Status']).lower() != 'done') or (str(row['Week']) == str(week_num) and str(row['Year']) == str(year))):
            line = (f"Account Manager: {row['AM']}\n"
                    f"Distributor: {row['Distributor']} | Country: {row['Country/Countries']} | Customers: {row['Customers']}\n"
                    f"Last question: {row['Question']}\n"
                    f"AM answer: {row['Comments / Feedback']}\n"
                    f"Status: {row['Status']}, Date: {row['Feedback Date']}\n"
                    "------")
            context_lines.append(line)
    return "\n".join(context_lines)

def extract_distributors_from_question(question):
    match = re.search(r"\(Distributor:\s*([^)]+)\)", question)
    if not match: return []
    dist_text = match.group(1)
    distributors = re.split(r",\s*|\s+and\s+", dist_text, flags=re.I)
    return [d.strip() for d in distributors if d.strip()]

def add_comparison_table_to_pdf(pdf, df):
    """
    Adds a single page with a consolidated comparison table to the PDF.
    - Colors percentage change values based on performance.
    """
    # === Define Time Periods ===
    today = datetime.today() + timedelta(days=11)
    current_week, current_year = today.isocalendar().week, today.isocalendar().year
    last_week_date = today - timedelta(weeks=1)
    last_week, last_week_year = last_week_date.isocalendar().week, last_week_date.isocalendar().year
    same_week_last_year_date = today - timedelta(weeks=52)
    same_week_last_year, same_week_last_year_year = same_week_last_year_date.isocalendar().week, same_week_last_year_date.isocalendar().year
    last_8_weeks = [(d.isocalendar().year, d.isocalendar().week) for d in [today - timedelta(weeks=i) for i in range(8)]]
    previous_8_weeks = [(d.isocalendar().year, d.isocalendar().week) for d in [today - timedelta(weeks=i) for i in range(8, 16)]]

    # === Filter and Aggregate Data ===
    current_week_df = df[df['YearWeek'] == (current_year, current_week)]
    last_week_df = df[df['YearWeek'] == (last_week_year, last_week)]
    same_week_last_year_df = df[df['YearWeek'] == (same_week_last_year_year, same_week_last_year)]
    last_8_weeks_df = df[df['YearWeek'].isin(last_8_weeks)]
    previous_8_weeks_df = df[df['YearWeek'].isin(previous_8_weeks)]
    
    # Aggregate data
    dfs_to_merge = [
        current_week_df.groupby('Product')['Total_mCi'].sum().rename('Current Wk'),
        last_week_df.groupby('Product')['Total_mCi'].sum().rename('Last Wk'),
        same_week_last_year_df.groupby('Product')['Total_mCi'].sum().rename('Same Wk LY'),
        last_8_weeks_df.groupby('Product')['Total_mCi'].sum().rename('Last 8 Wks'),
        previous_8_weeks_df.groupby('Product')['Total_mCi'].sum().rename('Prev 8 Wks')
    ]
    
    # Merge all aggregated data into one DataFrame
    merged_df = pd.concat(dfs_to_merge, axis=1).fillna(0)
    if merged_df.empty: return # Skip if no data
    
    # === Calculate Percentage Changes ===
    # Using a helper function to avoid division by zero errors
    def calc_pct_change(current, previous):
        return ((current - previous) / previous * 100).replace([np.inf, -np.inf], 100).fillna(0)

    merged_df['% Chg (WoW)'] = calc_pct_change(merged_df['Current Wk'], merged_df['Last Wk'])
    merged_df['% Chg (YoY)'] = calc_pct_change(merged_df['Current Wk'], merged_df['Same Wk LY'])
    merged_df['% Chg (8w)'] = calc_pct_change(merged_df['Last 8 Wks'], merged_df['Prev 8 Wks'])
    
    # === Prepare DataFrame for Display ===
    display_df = merged_df[[
        'Current Wk', 'Last Wk', '% Chg (WoW)',
        'Same Wk LY', '% Chg (YoY)',
        'Last 8 Wks', 'Prev 8 Wks', '% Chg (8w)'
    ]].reset_index()

    # Format numbers for display
    for col in ['Current Wk', 'Last Wk', 'Same Wk LY', 'Last 8 Wks', 'Prev 8 Wks']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:,.0f}')
    for col in ['% Chg (WoW)', '% Chg (YoY)', '% Chg (8w)']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:+.1f}%')

    # === Create the Matplotlib Table ===
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(display_df)) + 1))
    ax.axis('off')
    ax.set_title('Product Performance Comparison', fontsize=16, weight='bold', pad=20)
    
    col_labels = ['Product', 'Current Wk', 'Last Wk', '% Chg', 'Same Wk LY', '% Chg', 'Last 8 Wks', 'Prev 8 Wks', '% Chg']
    table = ax.table(cellText=display_df.values, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # === Apply Styling and Conditional Coloring ===
    pct_change_cols_indices = [3, 5, 8] # Column indices for % change columns
    
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.02
        if row == 0:
            cell.set_facecolor("#e6e6fa")
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#ffffff")
        
        # Color the percentage change values
        if col in pct_change_cols_indices and row > 0:
            # Get original numeric value for comparison
            numeric_value_col_name = merged_df.columns[col - 1] # -1 to map back to original df columns
            numeric_value = merged_df.iloc[row - 1][numeric_value_col_name]
            
            color = 'black'
            if numeric_value > 0.1: color = 'green'
            elif numeric_value < -0.1: color = 'red'
            cell.set_text_props(color=color, weight='bold')

        # Left-align the product name column
        if col == 0:
            cell.set_text_props(ha='left')
            cell.PAD = 0.04

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# --- Your utility/setup variables here ---
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

# === Add Account Manager mapping ===
account_manager_map = {
    "vbeillis@isotopia-global.com": "Vicki Beillis", "naricha@isotopia-global.com": "Noam Aricha",
    "ndellus@isotopia-global.com": "Noam Dellus", "gbader@isotopia-global.com": "Gilli Bader",
    "yfarchi@isotopia-global.com": "Yosi Farchi", "caksoy@isotopia-global.com": "Can Aksoy"
}
df["Account Manager"] = df["Account Manager Email"].map(account_manager_map).fillna("Other")

# === Define helper functions for APIs ===
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

# === Prepare weekly report data ===
today = datetime.today() + timedelta(days=11)
week_num, year = today.isocalendar().week, today.isocalendar().year
df = df.rename(columns={
    "The customer": "Customer", "Total amount ordered (mCi)": "Total_mCi", "Year": "Year",
    "Week of supply": "Week", "Shipping Status": "ShippingStatus",
    "Catalogue description (sold as)": "Product", "Distributing company (from Company name)": "Distributor", "Country": "Country"
})
valid_statuses = ["Shipped", "Partially shipped", "Shipped and arrived late", "Order being processed"]
df = df[df["ShippingStatus"].isin(valid_statuses)]
df = df.dropna(subset=["Customer", "Total_mCi", "Year", "Week", "ShippingStatus", "Product"])
for col in ["Total_mCi", "Year", "Week"]: df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Total_mCi", "Year", "Week"])
df["Year"] = df["Year"].astype(int); df["Week"] = df["Week"].astype(int)
df["YearWeek"] = list(zip(df["Year"], df["Week"]))

# === Build Mappings ===
distributor_country_to_customers = defaultdict(list); country_to_distributors = defaultdict(set)
country_to_customers = defaultdict(set); distributor_to_countries = defaultdict(set)
distributor_to_customers = defaultdict(set)
for _, row in df.iterrows():
    distributor = str(row['Distributor']).strip(); country = str(row['Country']).strip(); customer = str(row['Customer']).strip()
    if distributor and country and customer:
        distributor_country_to_customers[(distributor, country)].append(customer)
        country_to_distributors[country].add(distributor); country_to_customers[country].add(customer)
        distributor_to_countries[distributor].add(country); distributor_to_customers[distributor].add(customer)

# === Trends Analysis ===
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

# === Prepare content for prompts and summaries ===
def format_row_for_gpt(name, prev, curr, mgr):
    distributor = customer_to_distributor.get(name, "Unknown"); country = customer_to_country.get(name, "Unknown")
    return (f"{name:<30} | {curr - prev:+7.0f} mCi | "
            f"{(curr - prev) / prev * 100 if prev else 100:+6.1f}% | {mgr} | {distributor} | {country}")
gpt_rows = [format_row_for_gpt(name, prev, curr, mgr) for (name, prev, curr, mgr) in increased + decreased]
gpt_distributor_section = ["Customer name                     | Change   | % Change | Account Manager | Distributor | Country"] + gpt_rows
last_8_weeks = week_pairs[-8:]
previous_4_weeks, recent_4_weeks = last_8_weeks[:4], last_8_weeks[4:]
active_previous_4 = set(df[df["YearWeek"].isin(previous_4_weeks)]["Customer"])
active_recent_4 = set(df[df["YearWeek"].isin(recent_4_weeks)]["Customer"])
inactive_recent_4 = sorted(active_previous_4 - active_recent_4)
format_row = lambda name, prev, curr, mgr: (f"{name:<30} | {curr - prev:+7.0f} mCi | {(curr - prev) / prev * 100 if prev else 100:+6.1f}% | {mgr}", (curr - prev) / prev * 100 if prev else 100)
increased_formatted = sorted([format_row(*x) for x in increased], key=lambda x: x[1], reverse=True)
decreased_formatted = sorted([format_row(*x) for x in decreased], key=lambda x: x[1])
summary_lines = ["STOPPED ORDERING:", "...", "---"] + [f"{x[0]:<35} | {x[1]}" for x in stopped] if stopped else ["- None"]
summary_lines += ["", "DECREASED ORDERS:", "...", "---"] + [x[0] for x in decreased_formatted] if decreased_formatted else ["- None"]
summary_lines += ["", "INCREASED ORDERS:", "...", "---"] + [x[0] for x in increased_formatted] if increased_formatted else ["- None"]
summary_lines += ["", "INACTIVE IN PAST 4 WEEKS:", "...", "---"] + inactive_recent_4 if inactive_recent_4 else ["- None"]
summary_path = os.path.join(output_folder, "summary_test.txt")
with open(summary_path, "w") as f: f.write("\n".join(summary_lines))

# === ChatGPT Insights & Executive Summary Generation (Full block restored) ===
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key: raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=openai_api_key)
with open(summary_path, "r", encoding="utf-8") as f: report_text = f.read()
insight_history_path = os.path.join(output_folder, "insight_history_test.txt")
if os.path.exists(insight_history_path):
    with open(insight_history_path, "r", encoding="utf-8") as f: past_insights = f.read()
    report_text = f"{past_insights}\n\n===== NEW WEEK =====\n\n{report_text}"
feedback_context = build_feedback_context(feedback_df, week_num, year)
if not feedback_context.strip(): feedback_context = "No previous feedback available for this period."
full_prompt_text = ( "Previous feedback from Account Managers...\n" + feedback_context + "\n\n==== New Weekly Report Data ====\n\n" + report_text + "\n\n=== Distributor info for GPT analysis ===\n" + "\n".join(gpt_distributor_section))
system_prompt = "You are a senior business analyst..." # The long system prompt is unchanged
response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_prompt_text}])
insights = response.choices[0].message.content.strip()
print("\n💡 GPT Insights:\n", insights)
def extract_questions_by_am(insights):
    am_sections = re.split(r"\n([A-Z][a-z]+ [A-Z][a-z]+)\n", "\n" + insights)
    am_data = []
    for i in range(1, len(am_sections), 2):
        am_name = am_sections[i].strip()
        section = am_sections[i + 1]
        q_match = re.search(r"Questions\s*(for\s*[A-Za-z ]+)?\s*:\s*(.+?)(?=(\n[A-Z][a-z]+ [A-Z][a-z]+\n|$))", section, re.DOTALL)
        if not q_match: continue
        questions_text = q_match.group(2)
        for q_line in re.findall(r"\d+\.\s+(.+)", questions_text): am_data.append({"AM": am_name, "Question": q_line.strip()})
    return am_data
questions_by_am = extract_questions_by_am(insights)
new_rows = []
for q in questions_by_am:
    question_clean, distributors, countries, customers = extract_metadata_from_question(q['Question'])
    new_rows.append([week_num, year, q['AM'], ", ".join(distributors), ", ".join(countries), ", ".join(customers), question_clean, "", "Open", datetime.now().strftime("%Y-%m-%d")])
if new_rows: feedback_ws.append_rows(new_rows, value_input_option="USER_ENTERED"); print(f"✅ Added {len(new_rows)} new questions.")
else: print("No new questions found.")
exec_summary_prompt ="You are a senior business analyst..." # The exec summary prompt is unchanged
exec_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": exec_summary_prompt}, {"role": "user", "content": report_text}])
executive_summary = exec_response.choices[0].message.content.strip()
summary_json_path = os.path.join(output_folder, "Executive_Summary_test.json")
with open(summary_json_path, "w", encoding="utf-8") as f: json.dump({"executive_summary": executive_summary}, f, ensure_ascii=False, indent=2)
upload_to_drive(summary_json_path, "Executive_Summary_test.json", folder_id)
gsheet_service = build('sheets', 'v4', credentials=creds)
summary_spreadsheet_id = "185PnDw31D0zAYeUzNaSPQlQGTjmVVX1O9C86FC4JxV8"
gsheet_service.spreadsheets().values().clear(spreadsheetId=summary_spreadsheet_id, range="Executive Summary!A1").execute()
gsheet_service.spreadsheets().values().update(spreadsheetId=summary_spreadsheet_id, range="Executive Summary!A1", valueInputOption="RAW", body={"values": [[executive_summary]]}).execute()
with open(insight_history_path, "a") as f:
    f.write(f"\n\n===== Week {week_num}, {year} =====\n{insights}")


# === PDF Generation ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}_test.pdf")

with PdfPages(latest_pdf) as pdf:
    print("DEBUG: Entered PdfPages block")

    # --- Cover Page FIRST ---
    fig_cover = plt.figure(figsize=(9.5, 11))
    plt.axis("off")
    fig_cover.text(0.5, 0.80, "Weekly Orders Report", fontsize=26, ha="center", va="center", weight='bold')
    fig_cover.text(0.5, 0.74, f"Week {week_num}, {year}", fontsize=20, ha="center", va="center")
    logo_path = os.path.join(script_dir, "Isotopia.jpg")
    if os.path.exists(logo_path):
        logo = mpimg.imread(logo_path)
        ax_logo = fig_cover.add_axes([(1 - 0.35) / 2, 0.40, 0.35, 0.18])
        ax_logo.imshow(logo); ax_logo.axis("off")
    else: print("WARNING: Logo file not found, skipping logo.")
    pdf.savefig(fig_cover, bbox_inches="tight"); plt.close(fig_cover)
    
    # --- SECOND PAGE: Product Comparison Table ---
    add_comparison_table_to_pdf(pdf, df)

    # --- Rest of the report ---
    if not any([stopped, decreased, increased, inactive_recent_4]):
        fig_fallback = plt.figure(figsize=(8.5, 11)); plt.axis("off")
        fig_fallback.text(0.5, 0.5, "No trend data available for this week's report.", ha="center", va="center", fontsize=18)
        pdf.savefig(fig_fallback); plt.close(fig_fallback)
        print("⚠️ No trend data available. Generated fallback PDF with message page only.")
    else:
        # === Summary Tables (Full block restored) ===
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

        # === Top 5 Charts, Data Tables & Insights Pages (Full block restored) ===
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
        if stopped:
            stopped_df = pd.DataFrame([[name, wrap_text(customer_to_manager.get(name, "Other"))] for name, _ in sorted(stopped, key=lambda x: customer_to_manager.get(x[0], "Other"))], columns=["Customer", "Account Manager"])
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(stopped_df)) + 1)); ax.axis("off")
            table = ax.table(cellText=stopped_df.values, colLabels=stopped_df.columns, loc="upper left", cellLoc="left", colWidths=[0.8, 0.2])
            table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left")
            ax.set_title("STOPPED ORDERING", fontsize=14, weight="bold", pad=15); fig.text(0.5, 0.87, "Customers who stopped ordering in the last 4 weeks but did order in the 4 weeks before.", fontsize=10, ha="center")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        if decreased:
            for am, am_rows in sorted(decreased_by_am.items()):
                decreased_df = pd.DataFrame([[name, f"{curr - prev:+.0f}", f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%", wrap_text(am), (curr - prev) / prev * 100 if prev else 100] for name, prev, curr, _ in am_rows], columns=["Customer", "Change (mCi)", "% Change", "Account Manager", "PercentValue"]).sort_values("PercentValue").drop(columns="PercentValue")
                fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(decreased_df)) + 1)); ax.axis("off")
                table = ax.table(cellText=decreased_df.values, colLabels=decreased_df.columns, loc="upper left", cellLoc="center", colWidths=[0.64, 0.11, 0.11, 0.14])
                table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                    cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left" if col == 0 else "center")
                ax.set_title(f"DECREASED ORDERS – {am}", fontsize=14, weight="bold", pad=15); fig.text(0.5, 0.87, f"Customers managed by {am} who decreased orders in the last 8 weeks vs. prior 8.", fontsize=10, ha="center")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        if increased:
            for am, am_rows in sorted(increased_by_am.items()):
                increased_df = pd.DataFrame([[name, f"{curr - prev:+.0f}", f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%", wrap_text(am), (curr - prev) / prev * 100 if prev else 100] for name, prev, curr, _ in am_rows], columns=["Customer", "Change (mCi)", "% Change", "Account Manager", "PercentValue"]).sort_values("PercentValue", ascending=False).drop(columns="PercentValue")
                fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(increased_df)) + 1)); ax.axis("off")
                table = ax.table(cellText=increased_df.values, colLabels=increased_df.columns, loc="upper left", cellLoc="center", colWidths=[0.64, 0.11, 0.11, 0.14])
                table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                    cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left" if col == 0 else "center")
                ax.set_title(f"INCREASED ORDERS – {am}", fontsize=14, weight="bold", pad=15); fig.text(0.5, 0.87, f"Customers managed by {am} who increased orders in the last 8 weeks vs. prior 8.", fontsize=10, ha="center")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        if inactive_recent_4:
            inactive_df = pd.DataFrame([[name, wrap_text(customer_to_manager.get(name, "Other"))] for name in sorted(inactive_recent_4, key=lambda name: customer_to_manager.get(name, "Other"))], columns=["Customer", "Account Manager"])
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(inactive_df)) + 1)); ax.axis("off")
            table = ax.table(cellText=inactive_df.values, colLabels=inactive_df.columns, loc="upper left", cellLoc="left", colWidths=[0.8, 0.2])
            table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2; cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left")
            ax.set_title("INACTIVE IN PAST 4 WEEKS", fontsize=14, weight="bold", pad=15); fig.text(0.5, 0.87, "Customers who were active 4–8 weeks ago but not in the most recent 4 weeks.", fontsize=10, ha="center")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        insight_lines = [extract_metadata_from_question(line)[0] for line in insights.split("\n")]
        wrapped_insights = [item for line in insight_lines for item in textwrap.wrap(line, width=100, break_long_words=False)]
        lines_per_page = 35
        if not wrapped_insights:
            fig = plt.figure(figsize=(9.5, 11)); plt.axis("off"); fig.text(0.5, 0.5, "No insights available this week.", ha="center", fontsize=14); pdf.savefig(fig); plt.close(fig)
        else:
            for page_start in range(0, len(wrapped_insights), lines_per_page):
                fig = plt.figure(figsize=(9.5, 11)); plt.axis("off")
                page_lines = wrapped_insights[page_start:page_start + lines_per_page]
                fig.text(0.06, 0.95, "\n".join(page_lines), fontsize=10, ha="left", va="top")
                pdf.savefig(fig); plt.close(fig)

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
