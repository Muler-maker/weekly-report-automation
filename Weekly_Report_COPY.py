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
import numpy as np # Added for new comparison function
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

        # Remove metadata from question
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

def extract_distributors_from_question(question):
    match = re.search(r"\(Distributor:\s*([^)]+)\)", question)
    if not match:
        return []
    dist_text = match.group(1)
    distributors = re.split(r",\s*|\s+and\s+", dist_text, flags=re.I)
    return [d.strip() for d in distributors if d.strip()]

def add_comparison_tables_to_pdf(pdf, df):
    """
    Adds three comparison tables and their corresponding bar charts to the PDF.
    Args:
        pdf: The PdfPages object for the report.
        df: The main DataFrame containing the sales data.
    """
    # === Define Time Periods ===
    today = datetime.today() + timedelta(days=11) # Simulating the script's date logic
    current_week = today.isocalendar().week
    current_year = today.isocalendar().year

    last_week_date = today - timedelta(weeks=1)
    last_week = last_week_date.isocalendar().week
    last_week_year = last_week_date.isocalendar().year

    same_week_last_year_date = today - timedelta(weeks=52)
    same_week_last_year = same_week_last_year_date.isocalendar().week
    same_week_last_year_year = same_week_last_year_date.isocalendar().year

    # Last 8 weeks and previous 8 weeks
    last_8_weeks_dates = [today - timedelta(weeks=i) for i in range(8)]
    last_8_weeks = [(d.isocalendar().year, d.isocalendar().week) for d in last_8_weeks_dates]

    previous_8_weeks_dates = [today - timedelta(weeks=i) for i in range(8, 16)]
    previous_8_weeks = [(d.isocalendar().year, d.isocalendar().week) for d in previous_8_weeks_dates]

    # === Filter Data for Each Period ===
    current_week_df = df[(df['Year'] == current_year) & (df['Week'] == current_week)]
    last_week_df = df[(df['Year'] == last_week_year) & (df['Week'] == last_week)]
    same_week_last_year_df = df[(df['Year'] == same_week_last_year_year) & (df['Week'] == same_week_last_year)]
    
    # Ensure 'YearWeek' column exists for this comparison
    if 'YearWeek' not in df.columns:
        df["YearWeek"] = list(zip(df["Year"], df["Week"]))
        
    last_8_weeks_df = df[df['YearWeek'].isin(last_8_weeks)]
    previous_8_weeks_df = df[df['YearWeek'].isin(previous_8_weeks)]

    # === Aggregate Data by Product ===
    current_week_totals = current_week_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Current Week Total'})
    last_week_totals = last_week_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Last Week Total'})
    same_week_last_year_totals = same_week_last_year_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Same Week Last Year Total'})
    last_8_weeks_totals = last_8_weeks_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Last 8 Weeks Total'})
    previous_8_weeks_totals = previous_8_weeks_df.groupby('Product')['Total_mCi'].sum().reset_index().rename(columns={'Total_mCi': 'Previous 8 Weeks Total'})

    # === 1. Current Week vs. Last Week ===
    cw_vs_lw = pd.merge(current_week_totals, last_week_totals, on='Product', how='outer').fillna(0)
    if not cw_vs_lw.empty:
        cw_vs_lw['Change'] = cw_vs_lw['Current Week Total'] - cw_vs_lw['Last Week Total']
        cw_vs_lw['% Change'] = ((cw_vs_lw['Change'] / cw_vs_lw['Last Week Total']) * 100).replace([np.inf, -np.inf], 100).fillna(0)

    # === 2. Current Week vs. Same Week Last Year ===
    cw_vs_swly = pd.merge(current_week_totals, same_week_last_year_totals, on='Product', how='outer').fillna(0)
    if not cw_vs_swly.empty:
        cw_vs_swly['Change'] = cw_vs_swly['Current Week Total'] - cw_vs_swly['Same Week Last Year Total']
        cw_vs_swly['% Change'] = ((cw_vs_swly['Change'] / cw_vs_swly['Same Week Last Year Total']) * 100).replace([np.inf, -np.inf], 100).fillna(0)

    # === 3. Last 8 Weeks vs. Previous 8 Weeks ===
    l8w_vs_p8w = pd.merge(last_8_weeks_totals, previous_8_weeks_totals, on='Product', how='outer').fillna(0)
    if not l8w_vs_p8w.empty:
        l8w_vs_p8w['Change'] = l8w_vs_p8w['Last 8 Weeks Total'] - l8w_vs_p8w['Previous 8 Weeks Total']
        l8w_vs_p8w['% Change'] = ((l8w_vs_p8w['Change'] / l8w_vs_p8w['Previous 8 Weeks Total']) * 100).replace([np.inf, -np.inf], 100).fillna(0)

    # --- Plotting Function ---
    def plot_comparison(df, title, columns, pdf_pages):
        if df.empty or df[columns].sum().sum() == 0:
            print(f"Skipping empty plot: {title}")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot = df.set_index('Product')[columns]
        df_plot.plot(kind='bar', ax=ax)
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_ylabel('Total mCi')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)

    # --- Add to PDF ---
    plot_comparison(cw_vs_lw, 'Bars comparison: total amounts per product\n Current week vs. Last week', ['Current Week Total', 'Last Week Total'], pdf)
    plot_comparison(cw_vs_swly, 'Bars comparison: total amounts per product\n Current week vs. Same week last year', ['Current Week Total', 'Same Week Last Year Total'], pdf)
    plot_comparison(l8w_vs_p8w, 'Bars comparison: total amounts per product\n Last 8 weeks vs. Previous 8 weeks', ['Last 8 Weeks Total', 'Previous 8 Weeks Total'], pdf)


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
# This is redundant if already done, but harmless
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
    "gbader@isotopia-global.com": "Gilli Bader",
    "yfarchi@isotopia-global.com": "Yosi Farchi",
    "caksoy@isotopia-global.com": "Can Aksoy"
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
        }

        response = requests.post("https://api.resend.com/emails", data=data, files=files, headers=headers)

    if 200 <= response.status_code < 300:
        print(f"📨 Email successfully sent to: {', '.join(to_emails)}")
    else:
        print("❌ Failed to send email:", response.status_code, response.text)

def upload_to_drive(file_path, file_name, folder_id=None):
    drive_service = build('drive', 'v3', credentials=creds)
    query = f"name='{file_name}' and trashed=false"
    if folder_id:
        query += f" and '{folder_id}' in parents"

    existing_files = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute().get('files', [])
    
    for file in existing_files:
        drive_service.files().delete(fileId=file['id']).execute()
        print(f"🗑️ Deleted: {file['name']} (ID: {file['id']})")

    file_metadata = {'name': file_name, 'parents': [folder_id] if folder_id else []}
    mimetype = 'application/json' if file_name.endswith('.json') else 'application/pdf'
    if file_name.endswith('.txt'):
        mimetype = 'text/plain'

    media = MediaFileUpload(file_path, mimetype=mimetype)
    uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
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

# === Build Mappings ===
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

increased_by_am = defaultdict(list)
decreased_by_am = defaultdict(list)

for row in increased:
    increased_by_am[row[3]].append(row)

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

gpt_rows = [format_row_for_gpt(name, prev, curr, mgr) for (name, prev, curr, mgr) in increased + decreased]
gpt_distributor_section = ["Customer name                     | Change   | % Change | Account Manager | Distributor | Country"] + gpt_rows

# === Inactive recent 4 weeks ===
last_8_weeks = week_pairs[-8:]
previous_4_weeks, recent_4_weeks = last_8_weeks[:4], last_8_weeks[4:]
active_previous_4 = set(df[df["YearWeek"].isin(previous_4_weeks)]["Customer"])
active_recent_4 = set(df[df["YearWeek"].isin(recent_4_weeks)]["Customer"])
inactive_recent_4 = sorted(active_previous_4 - active_recent_4)

# === Format summary text ===
format_row = lambda name, prev, curr, mgr: (f"{name:<30} | {curr - prev:+7.0f} mCi | {(curr - prev) / prev * 100 if prev else 100:+6.1f}% | {mgr}", (curr - prev) / prev * 100 if prev else 100)
increased_formatted = sorted([format_row(*x) for x in increased], key=lambda x: x[1], reverse=True)
decreased_formatted = sorted([format_row(*x) for x in decreased], key=lambda x: x[1])

summary_lines = [
    "STOPPED ORDERING:",
    "Customers who stopped ordering in the last 8 weeks but did order in the 8 weeks before.",
    "Customer Name                     | Account Manager",
    "---------------------------------------------------"
] + [f"{x[0]:<35} | {x[1]}" for x in stopped] if stopped else ["- None"]

summary_lines += ["", "DECREASED ORDERS:", "These customers ordered less in the last 8 weeks compared to the 8 weeks prior.", "Customer Name                     | Change   | % Change | Account Manager", "--------------------------------------------------------------------------"] + [x[0] for x in decreased_formatted] if decreased_formatted else ["- None"]
summary_lines += ["", "INCREASED ORDERS:", "These customers increased their order amounts in the last 8 weeks compared to the 8 weeks prior.", "Customer Name                     | Change   | % Change | Account Manager", "--------------------------------------------------------------------------"] + [x[0] for x in increased_formatted] if increased_formatted else ["- None"]
summary_lines += ["", "INACTIVE IN PAST 4 WEEKS:", "Customers who ordered in the previous 4 weeks but not in the most recent 4.", "Customer Name", "-------------------------------"] + inactive_recent_4 if inactive_recent_4 else ["- None"]

# === Save summary ===
summary_path = os.path.join(output_folder, "summary_test.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

# === ChatGPT Insights with memory ===
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=openai_api_key)

with open(summary_path, "r", encoding="utf-8") as f:
    report_text = f.read()

insight_history_path = os.path.join(output_folder, "insight_history_test.txt")
if os.path.exists(insight_history_path):
    with open(insight_history_path, "r", encoding="utf-8") as f:
        past_insights = f.read()
    report_text = f"{past_insights}\n\n===== NEW WEEK =====\n\n{report_text}"

feedback_context = build_feedback_context(feedback_df, week_num, year)
if not feedback_context.strip():
    feedback_context = "No previous feedback available for this period."

full_prompt_text = ("Previous feedback from Account Managers (from the Feedback loop sheet):\n" + feedback_context + "\n\n==== New Weekly Report Data ====\n\n" + report_text + "\n\n=== Distributor info for GPT analysis ===\n" + "\n".join(gpt_distributor_section))

system_prompt = """
You are a senior business analyst. Analyze the weekly report for each Account Manager using a top-down structure: Distributor-level trends, followed by country-level patterns, and then customer-specific insights.

For each Account Manager:
Start their section with their full name on a line by itself (e.g., Vicki Beillis).
If there are no significant trends or issues for that week, write: No significant insights or questions for this week.
Otherwise, provide a concise summary of relevant trends and insights, grouped as follows:
Distributor-level: Note important patterns, risks, or deviations, including expected order behaviors.
Country-level: Summarize trends affecting several customers in the same country.
Customer-specific: Highlight notable changes, spikes, drops, or inactivity at the customer level.

After the insights for each Account Manager, always include a separate section titled exactly as follows:

Questions for [Account Manager Name]:
Provide up to 2 questions per Account Manager. If fewer than 2 relevant questions exist, include only those.
[First question] (Distributor: [Distributor Name(s)]; Country: [Country Name(s)]; Customer: [Customer Name(s)])
[Second question] (Distributor: [Distributor Name(s)]; Country: [Country Name(s)]; Customer: [Customer Name(s)])
[Third question] (Distributor: [Distributor Name(s)]; Country: [Country Name(s)]; Customer: [Customer Name(s)])
(Use numbered questions, one per line, and use the AM’s exact name in the heading. Each question must explicitly specify the relevant distributor(s), country or countries, and customer(s) in parentheses exactly as shown.)

Avoiding Redundant Questions:
Do not repeat questions that were already asked and fully answered in recent weeks unless a new deviation or anomaly is observed.
If a customer's ordering behavior remains consistent with a previously confirmed explanation, acknowledge that no follow-up is required.

Metadata specification instructions:
- Every question must include a metadata block, exactly as shown:
  (Distributor: [Distributor Name(s)]; Country: [Country Name(s)]; Customer: [Customer Name(s)])
- All three fields—Distributor, Country, Customer—must always be filled. Never leave a field blank.
  If a field is not directly mentioned in the question, infer or expand as follows (using the provided data):
    - If Distributor and Country are given, but not Customer:
        List all customers linked to that distributor in that country, separated by commas.
    - If only Country is given:
        List all distributors and all customers in that country.
    - If only Distributor is given:
        List all countries and all customers for that distributor.
    - If Customer is missing or matches Distributor:
        Fill both fields with that value.
    - If there is no valid value, use "N/A".
    - Separate multiple values with commas in each field.
- Use semicolons to separate the three metadata fields.
- Use this exact metadata format for every question.

Example Formatting:

Question:
What might explain the decrease in orders from Sinotau Pharmaceutical Group in China?
Metadata:
(Distributor: Sinotau Pharmaceutical Group; Country: China; Customer: Sinotau Pharmaceutical Group (Guangdong), Sinotau Pharmaceutical Group (Beijing))

Question:
Are there general order trends in Germany this month?
Metadata:
(Distributor: DSD Pharma GmbH, ABC Distributors; Country: Germany; Customer: University Hospital, Berlin Clinic)

Question:
Is Sinotau Pharmaceutical Group maintaining order volumes globally?
Metadata:
(Distributor: Sinotau Pharmaceutical Group; Country: China, Singapore; Customer: Sinotau (Guangdong), Sinotau (Beijing), Sinotau (Singapore))

Question:
Is COMISSÃO NACIONAL DE ENERGIA NUCLEAR (CNEN) maintaining their expected schedule?
Metadata:
(Distributor: COMISSÃO NACIONAL DE ENERGIA NUCLEAR (CNEN); Country: Brazil; Customer: COMISSÃO NACIONAL DE ENERGIA NUCLEAR (CNEN))

Additional instructions on specifying customers and countries in questions:
- When a customer is different from the distributor, specify the customer’s name and country in the question body, phrased like:
  "the [Customer Name] customer in [Country]"
  followed by "of distributor [Distributor Name(s)]".
- If the customer and distributor are the same entity, omit the Customer field in the body but always fill it in the metadata.
- Ensure countries correspond to the customer(s) mentioned.

Guidelines:
- Base your questions on both the current report and the most recent feedback or answers from previous cycles.
- For ongoing or unresolved issues, ask clarifying or follow-up questions and reference previous feedback where relevant.
- For new issues, ask investigative questions to help clarify the root cause or suggest possible next steps.
- Reference previous reports and feedback to highlight new, ongoing, or resolved issues.
- Present only insights, trends, and questions—do not include recommendations or action items.
- Use only plain text. No Markdown, asterisks, or any special formatting.
- COMISSÃO NACIONAL DE ENERGIA NUCLEAR (CNEN) is expected to order every two weeks on even-numbered weeks. Flag and ask about any deviation from this pattern.

Example:
No follow-up required for Global Medical Solutions Australia – order behavior remains consistent with ANSTO-related supply gaps.
Avoid re-asking questions about the following customers unless a new pattern emerges:
St. Luke’s Medical Center INC. (Philippines) – ongoing onboarding and evaluation phase
Global Medical Solutions Australia – orders only placed during ANSTO shutdowns
UMC Utrecht (Netherlands) – seasonal holiday slowdown
Evergreen Theragnostics (USA) – clinical trial and internal CDMO usage
Do not generate a new question for these unless there is a change in behavior, such as missed expected weeks, larger-than-expected volumes, or extended inactivity

How to interpret the table of previous questions:
You are provided with a table containing past questions, feedback, and resolution status.
Use this table to inform your analysis and avoid repeating previously answered questions.

Follow these rules:
Only treat rows marked "Close" as resolved. Do not repeat the question unless current behavior contradicts the previous explanation.
If behavior is consistent with the prior answer, add:
No follow-up required for [Customer] – behavior remains consistent with previous feedback.
For rows marked "Open", assume the issue remains unresolved. You may:
Re-ask the original question
Ask a follow-up to prompt clarification
Mention the issue has been open since its feedback date
Prioritize open issues that have remained unresolved for multiple weeks.

Column meanings:
Customers – one or more customer names, possibly comma-separated
Question – the original inquiry
Comments / Feedback – the response provided by the team
Status – either “Open” (unresolved) or “Close” (resolved)
Feedback Date – when the status or answer was last updated
Week Number – optional; helps track how long issues remain unresolved
"""

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
        if not q_match:
            continue
        questions_text = q_match.group(2)
        for q_line in re.findall(r"\d+\.\s+(.+)", questions_text):
            am_data.append({"AM": am_name, "Question": q_line.strip()})
    return am_data

questions_by_am = extract_questions_by_am(insights)
new_rows = []
for q in questions_by_am:
    question_clean, distributors, countries, customers = extract_metadata_from_question(q['Question'])
    new_rows.append([week_num, year, q['AM'], ", ".join(distributors), ", ".join(countries), ", ".join(customers), question_clean, "", "Open", datetime.now().strftime("%Y-%m-%d")])

if new_rows:
    feedback_ws.append_rows(new_rows, value_input_option="USER_ENTERED")
    print(f"✅ Added {len(new_rows)} new questions to Feedback loop worksheet.")
else:
    print("No new questions found to add to the Feedback loop worksheet.")

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

exec_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": exec_summary_prompt}, {"role": "user", "content": report_text}])
executive_summary = exec_response.choices[0].message.content.strip()

summary_json_path = os.path.join(output_folder, "Executive_Summary_test.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump({"executive_summary": executive_summary}, f, ensure_ascii=False, indent=2)

upload_to_drive(summary_json_path, "Executive_Summary_test.json", folder_id)

gsheet_service = build('sheets', 'v4', credentials=creds)
summary_spreadsheet_id = "185PnDw31D0zAYeUzNaSPQlQGTjmVVX1O9C86FC4JxV8"
sheet_metadata = gsheet_service.spreadsheets().get(spreadsheetId=summary_spreadsheet_id).execute()
sheet_titles = [s['properties']['title'] for s in sheet_metadata['sheets']]
if "Executive Summary" not in sheet_titles:
    requests_body = {"requests": [{"addSheet": {"properties": {"title": "Executive Summary"}}}]}
    gsheet_service.spreadsheets().batchUpdate(spreadsheetId=summary_spreadsheet_id, body=requests_body).execute()

gsheet_service.spreadsheets().values().clear(spreadsheetId=summary_spreadsheet_id, range="Executive Summary!A1").execute()
gsheet_service.spreadsheets().values().update(spreadsheetId=summary_spreadsheet_id, range="Executive Summary!A1", valueInputOption="RAW", body={"values": [[executive_summary]]}).execute()
print("✅ Executive Summary successfully written into Google Sheet.")

with open(insight_history_path, "a") as f:
    f.write(f"\n\n===== Week {week_num}, {year} =====\n")
    f.write(insights)

# === PDF Generation ===
latest_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Week_{week_num}_{year}_test.pdf")

with PdfPages(latest_pdf) as pdf:
    print("DEBUG: Entered PdfPages block")

    # --- Add the new comparison tables at the beginning ---
    add_comparison_tables_to_pdf(pdf, df)

    if not any([stopped, decreased, increased, inactive_recent_4]):
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        fig.text(0.5, 0.5, "No data available for this week's report.", ha="center", va="center", fontsize=18)
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
            ax_logo = fig.add_axes([(1 - 0.35) / 2, 0.40, 0.35, 0.18])
            ax_logo.imshow(logo)
            ax_logo.axis("off")
        else:
            print("WARNING: Logo file not found, skipping logo.")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # === Summary Table: Overall Categories ===
        total_decreased = sum(curr - prev for v in decreased_by_am.values() for _, prev, curr, _ in v)
        total_increased = sum(curr - prev for v in increased_by_am.values() for _, prev, curr, _ in v)

        summary_df = pd.DataFrame([["Stopped Ordering", len(stopped), "–"], ["Decreased Orders", sum(len(v) for v in decreased_by_am.values()), f"{total_decreased:+.1f}"], ["Increased Orders", sum(len(v) for v in increased_by_am.values()), f"{total_increased:+.1f}"], ["Inactive (last 4 wks)", len(inactive_recent_4), "–"]], columns=["Category", "# of Customers", "Total Δ (mCi)"])
        fig, ax = plt.subplots(figsize=(8.5, max(2.2, 0.4 + 0.3 * len(summary_df))))
        ax.axis("off")
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, loc="center", cellLoc="center")
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
        am_list = set(list(decreased_by_am.keys()) + list(increased_by_am.keys()) + [customer_to_manager.get(name, "Other") for name in inactive_recent_4 + [x[0] for x in stopped]])
        am_summary = []
        for am in sorted(am_list):
            am_summary.append([am, len(increased_by_am.get(am, [])), len(decreased_by_am.get(am, [])), sum(1 for name in inactive_recent_4 if customer_to_manager.get(name, "Other") == am), sum(1 for name, _ in stopped if customer_to_manager.get(name, "Other") == am)])
        am_df = pd.DataFrame(am_summary, columns=["Account Manager", "+ Customers", "– Customers", "Inactive", "Stopped"])
        fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.4 + 0.3 * len(am_df))))
        ax.axis("off")
        table = ax.table(cellText=am_df.values, colLabels=am_df.columns, loc="center", cellLoc="center")
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
        products = {"Lutetium  (177Lu) chloride N.C.A.": "Top 5 N.C.A. Customers", "Lutetium (177Lu) chloride C.A": "Top 5 C.A. Customers", "Terbium-161 chloride n.c.a": "Top 5 Terbium Customers"}
        for product_name, title in products.items():
            product_df = recent_df[recent_df["Product"] == product_name]
            if product_df.empty: continue
            top_customers = product_df.groupby("Customer")["Total_mCi"].sum().nlargest(5).index
            plot_df = product_df[product_df["Customer"].isin(top_customers)].copy()
            plot_df["WrappedCustomer"] = plot_df["Customer"].apply(lambda x: '\n'.join(textwrap.wrap(x, 12)))
            plot_df["WeekLabel"] = plot_df["Year"].astype(str) + "-W" + plot_df["Week"].astype(str).str.zfill(2)
            pivot_df = plot_df.pivot_table(index="WeekLabel", columns="WrappedCustomer", values="Total_mCi", aggfunc="sum").fillna(0)
            pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=lambda x: (int(x.split("-W")[0]), int(x.split("-W")[1]))))
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

        # --- Data Tables (Stopped, Decreased, Increased, Inactive) ---
        # (This part of the code remains the same as your original script for generating these tables)
        if stopped:
            stopped_sorted = sorted(stopped, key=lambda x: customer_to_manager.get(x[0], "Other"))
            stopped_df = pd.DataFrame([[name, wrap_text(customer_to_manager.get(name, "Other"))] for name, _ in stopped_sorted], columns=["Customer", "Account Manager"])
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(stopped_df)) + 1))
            ax.axis("off")
            table = ax.table(cellText=stopped_df.values, colLabels=stopped_df.columns, loc="upper left", cellLoc="left", colWidths=[0.8, 0.2])
            table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2
                cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left")
            ax.set_title("STOPPED ORDERING", fontsize=14, weight="bold", pad=15)
            fig.text(0.5, 0.87, "Customers who stopped ordering in the last 4 weeks but did order in the 4 weeks before.", fontsize=10, ha="center")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if decreased:
            for am, am_rows in sorted(decreased_by_am.items()):
                decreased_df = pd.DataFrame([[name, f"{curr - prev:+.0f}", f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%", wrap_text(am), (curr - prev) / prev * 100 if prev else 100] for name, prev, curr, _ in am_rows], columns=["Customer", "Change (mCi)", "% Change", "Account Manager", "PercentValue"]).sort_values("PercentValue").drop(columns="PercentValue")
                fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(decreased_df)) + 1))
                ax.axis("off")
                table = ax.table(cellText=decreased_df.values, colLabels=decreased_df.columns, loc="upper left", cellLoc="center", colWidths=[0.64, 0.11, 0.11, 0.14])
                table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2
                    cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                    cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left" if col == 0 else "center")
                ax.set_title(f"DECREASED ORDERS – {am}", fontsize=14, weight="bold", pad=15)
                fig.text(0.5, 0.87, f"Customers managed by {am} who decreased orders in the last 8 weeks vs. prior 8.", fontsize=10, ha="center")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if increased:
            for am, am_rows in sorted(increased_by_am.items()):
                increased_df = pd.DataFrame([[name, f"{curr - prev:+.0f}", f"{(curr - prev) / prev * 100:+.1f}%" if prev else "+100%", wrap_text(am), (curr - prev) / prev * 100 if prev else 100] for name, prev, curr, _ in am_rows], columns=["Customer", "Change (mCi)", "% Change", "Account Manager", "PercentValue"]).sort_values("PercentValue", ascending=False).drop(columns="PercentValue")
                fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(increased_df)) + 1))
                ax.axis("off")
                table = ax.table(cellText=increased_df.values, colLabels=increased_df.columns, loc="upper left", cellLoc="center", colWidths=[0.64, 0.11, 0.11, 0.14])
                table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2
                    cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                    cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left" if col == 0 else "center")
                ax.set_title(f"INCREASED ORDERS – {am}", fontsize=14, weight="bold", pad=15)
                fig.text(0.5, 0.87, f"Customers managed by {am} who increased orders in the last 8 weeks vs. prior 8.", fontsize=10, ha="center")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if inactive_recent_4:
            inactive_sorted = sorted(inactive_recent_4, key=lambda name: customer_to_manager.get(name, "Other"))
            inactive_df = pd.DataFrame([[name, wrap_text(customer_to_manager.get(name, "Other"))] for name in inactive_sorted], columns=["Customer", "Account Manager"])
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 + 0.3 * len(inactive_df)) + 1))
            ax.axis("off")
            table = ax.table(cellText=inactive_df.values, colLabels=inactive_df.columns, loc="upper left", cellLoc="left", colWidths=[0.8, 0.2])
            table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.2
                cell.set_facecolor("#e6e6fa" if row == 0 else ("#f9f9f9" if row % 2 == 0 else "#ffffff"))
                cell.set_text_props(weight='bold' if row == 0 else 'normal', ha="left")
            ax.set_title("INACTIVE IN PAST 4 WEEKS", fontsize=14, weight="bold", pad=15)
            fig.text(0.5, 0.87, "Customers who were active 4–8 weeks ago but not in the most recent 4 weeks.", fontsize=10, ha="center")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # === ChatGPT Insights Pages ===
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

print("DEBUG: PDF file size right after creation:", os.path.getsize(latest_pdf))

# === Save and Upload Reports with _test suffix ===
summary_pdf = os.path.join(output_folder, f"Weekly_Orders_Report_Summary_Week_{week_num}_{year}_test.pdf")
latest_copy_path = os.path.join(output_folder, "Latest_Weekly_Report_test.pdf")
copyfile(latest_pdf, summary_pdf)
copyfile(latest_pdf, latest_copy_path)
print(f"✅ Report also saved as: {summary_pdf}")
print(f"✅ Report also saved as: {latest_copy_path}")

time.sleep(0.5)

week_info_path = os.path.join(output_folder, "Week_number_test.txt")
with open(week_info_path, "w") as f:
    f.write(f"{week_num},{year}")

upload_to_drive(summary_pdf, f"Weekly_Orders_Report_Summary_Week_{week_num}_{year}_test.pdf", folder_id)
upload_to_drive(latest_copy_path, "Latest_Weekly_Report_test.pdf", folder_id)
upload_to_drive(week_info_path, "Week_number_test.txt", folder_id)
