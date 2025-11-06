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

def add_comparison_page_to_pdf(pdf, df):
    """
    Adds a single page with three comparison bar charts to the PDF.
    - Applies custom color palettes for product types.
    """
    # === Plotting Helper with Custom Colors ===
    def plot_comparison_on_axis(ax, df, title, columns):
        if df.empty or df[columns].sum().sum() == 0:
            ax.text(0.5, 0.5, 'No data for this comparison', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, weight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            return

        df_plot = df.set_index('Product')[columns]
        df_plot.plot(kind='bar', ax=ax, legend=False, width=0.8) # No default legend

        # --- Custom Coloring Logic ---
        color_map = {'N.C.A.': '#FFA500', 'C.A': '#1f77b4', 'Terbium': '#2ca02c'}
        default_color = 'grey'
        
        product_labels = [label.get_text() for label in ax.get_xticklabels()]
        num_groups = len(df_plot.columns)
        num_products = len(product_labels)

        for i, product_name in enumerate(product_labels):
            color = default_color
            if 'N.C.A.' in product_name: color = color_map['N.C.A.']
            elif 'C.A' in product_name: color = color_map['C.A']
            elif 'Terbium' in product_name: color = color_map['Terbium']

            for j in range(num_groups):
                patch_index = j * num_products + i
                if patch_index < len(ax.patches):
                    ax.patches[patch_index].set_facecolor(color)

        # --- Custom Legend ---
        legend_elements = [Patch(facecolor=color_map['N.C.A.'], label='Lutetium N.C.A.'),
                           Patch(facecolor=color_map['C.A'], label='Lutetium C.A.'),
                           Patch(facecolor=color_map['Terbium'], label='Terbium'),
                           Patch(facecolor=default_color, label='Other')]
        ax.legend(handles=legend_elements, title="Product Type", fontsize='small')

        ax.set_title(title, fontsize=12, weight='bold', pad=15)
        ax.set_ylabel('Total mCi', fontsize=9)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=80, labelsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

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
    current_week_df = df[(df['Year'] == current_year) & (df['Week'] == current_week)]
    last_week_df = df[(df['Year'] == last_week_year) & (df['Week'] == last_week)]
    same_week_last_year_df = df[(df['Year'] == same_week_last_year_year) & (df['Week'] == same_week_last_year)]
    if 'YearWeek' not in df.columns: df["YearWeek"] = list(zip(df["Year"], df["Week"]))
    last_8_weeks_df = df[df['YearWeek'].isin(last_8_weeks)]
    previous_8_weeks_df = df[df['YearWeek'].isin(previous_8_weeks)]

    current_week_totals = current_week_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Current Week'})
    last_week_totals = last_week_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Last Week'})
    same_week_last_year_totals = same_week_last_year_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Same Week Last Year'})
    last_8_weeks_totals = last_8_weeks_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Last 8 Weeks'})
    previous_8_weeks_totals = previous_8_weeks_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Previous 8 Weeks'})

    # === Create Comparison DataFrames ===
    cw_vs_lw = pd.merge(current_week_totals, last_week_totals, on='Product', how='outer').fillna(0)
    cw_vs_swly = pd.merge(current_week_totals, same_week_last_year_totals, on='Product', how='outer').fillna(0)
    l8w_vs_p8w = pd.merge(last_8_weeks_totals, previous_8_weeks_totals, on='Product', how='outer').fillna(0)

    # === Create the Single Page Figure ===
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5, 11))
    fig.suptitle('Product Performance Comparison', fontsize=16, weight='bold')

    plot_comparison_on_axis(ax1, cw_vs_lw, 'Current Week vs. Last Week', ['Current Week', 'Last Week'])
    plot_comparison_on_axis(ax2, cw_vs_swly, 'Current Week vs. Same Week Last Year', ['Current Week', 'Same Week Last Year'])
    plot_comparison_on_axis(ax3, l8w_vs_p8w, 'Last 8 Weeks vs. Previous 8 Weeks', ['Last 8 Weeks', 'Previous 8 Weeks'])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

# --- Your utility/setup variables here ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1uJAArBkHXgFvY_JPIieKRjiOYd9-Ys7Ii9nXT3fWbUg"

# === Authenticate Google Sheets (do this before loading sheets) ===
SERVICE_ACCOUNT_FILE = os.path.join(script_dir, 'credentials.json')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
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

# === Load data from Google Sheets ===
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1WNW4IqAG6W_6XVSaSk9kkukztY93qCDQ3mypZaF5F-Y"
sheet = gc.open_by_url(SPREADSHEET_URL).worksheet("Airtable Data")
df = pd.DataFrame(sheet.get_all_records())
df.columns = [c.strip() for c in df.columns]

# === Add Account Manager mapping ===
account_manager_map = {
    "vbeillis@isotopia-global.com": "Vicki Beillis", "naricha@isotopia-global.com": "Noam Aricha",
    "ndellus@isotopia-global.com": "Noam Dellus", "gbader@isotopia-global.com": "Gilli Bader",
    "yfarchi@isotopia-global.com": "Yosi Farchi", "caksoy@isotopia-global.com": "Can Aksoy"
}
df["Account Manager"] = df["Account Manager Email"].map(account_manager_map).fillna("Other")

# === Define email sending function using Resend API ===
def send_email(subject, body, to_emails, attachment_path):
    # ... (function remains the same)

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

# === Define report date values (simulate 11 days ahead) ===
today = datetime.today() + timedelta(days=11)
week_num, year = today.isocalendar().week, today.isocalendar().year

# === Prepare weekly report data ===
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

# === Define 16-week window ===
current = today - timedelta(days=today.weekday())
week_pairs = [(d.isocalendar().year, d.isocalendar().week) for d in [current - timedelta(weeks=(15 - i)) for i in range(16)]]
previous_8, recent_8 = week_pairs[:8], week_pairs[8:]
recent_df = df[df["YearWeek"].isin(recent_8)]; previous_df = df[df["YearWeek"].isin(previous_8)]

# === Trends Analysis ===
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
# ... (this section remains the same)
def format_row_for_gpt(name, prev, curr, mgr):
    distributor = customer_to_distributor.get(name, "Unknown")
    country = customer_to_country.get(name, "Unknown")
    return (f"{name:<30} | {curr - prev:+7.0f} mCi | "
            f"{(curr - prev) / prev * 100 if prev else 100:+6.1f}% | "
            f"{mgr} | {distributor} | {country}")
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
summary_lines = ["STOPPED ORDERING:", "...", "---------------------------------------------------"] + [f"{x[0]:<35} | {x[1]}" for x in stopped] if stopped else ["- None"]
summary_lines += ["", "DECREASED ORDERS:", "...", "--------------------------------------------------------------------------"] + [x[0] for x in decreased_formatted] if decreased_formatted else ["- None"]
summary_lines += ["", "INCREASED ORDERS:", "...", "--------------------------------------------------------------------------"] + [x[0] for x in increased_formatted] if increased_formatted else ["- None"]
summary_lines += ["", "INACTIVE IN PAST 4 WEEKS:", "...", "-------------------------------"] + inactive_recent_4 if inactive_recent_4 else ["- None"]
summary_path = os.path.join(output_folder, "summary_test.txt")
with open(summary_path, "w") as f: f.write("\n".join(summary_lines))


# === ChatGPT Insights with memory ===
# ... (This entire section for interacting with OpenAI API remains the same)
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
# ... (rest of the script for writing to google sheets and history file remains the same)
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
        ax_logo.imshow(logo)
        ax_logo.axis("off")
    else:
        print("WARNING: Logo file not found, skipping logo.")
    pdf.savefig(fig_cover, bbox_inches="tight")
    plt.close(fig_cover)
    
    # --- SECOND PAGE: Product Comparison Charts ---
    add_comparison_page_to_pdf(pdf, df)

    # --- Rest of the report ---
    if not any([stopped, decreased, increased, inactive_recent_4]):
        fig_fallback = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        fig_fallback.text(0.5, 0.5, "No trend data available for this week's report.", ha="center", va="center", fontsize=18)
        pdf.savefig(fig_fallback)
        plt.close(fig_fallback)
        print("⚠️ No trend data available. Generated fallback PDF with message page only.")
    else:
        # === Summary Table: Overall Categories ===
        # ... (rest of the PDF generation code is unchanged)
        # === Summary Table: By Account Manager ===
        # ...
        # === Add Top 5 Charts by Product ===
        # ...
        # --- Data Tables (Stopped, Decreased, Increased, Inactive) ---
        # ...
        # === ChatGPT Insights Pages ===
        # ...
        pass # Placeholder for the rest of your original, unchanged PDF generation logic

# ... (The final part of your script for saving and uploading remains the same)
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
