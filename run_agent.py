import os
import time
import json
import feedparser
import requests
import pandas as pd
from bs4 import BeautifulSoup
from google.oauth2.service_account import Credentials
import gspread
from openai import OpenAI


# ---------------------- CONFIG ----------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RSS_FILE = os.getenv("RSS_FILE", "feeds_rss.txt")
HOSPITAL_FILE = os.getenv("HOSPITAL_FILE", "hospitals.txt")
OUT_CSV = os.getenv("OUT_CSV", "tenders_lu177_tb161.csv")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
TAB = os.getenv("GOOGLE_SHEET_TAB", "Sheet1")

# Google credentials
SA_JSON = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(SA_JSON, scopes=scopes)
gc = gspread.authorize(creds)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------- HELPERS ----------------------
def summarize_text(text: str):
    """Use GPT to check if text relates to Lu-177 or Tb-161 tenders."""
    prompt = f"""
You are a research assistant. Read the following tender excerpt.
Decide if it is relevant to the supply or procurement of Lutetium-177 or Terbium-161 (radiopharmaceutical isotopes).
Respond with a JSON object: 
{{
  "relevance": 1 if clearly related else 0,
  "confidence": number between 0 and 1,
  "summary": short summary (English)
}}
Text:
{text}
"""
    try:
        resp = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=250
        )
        raw = resp.output_text
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        data = {"relevance": 0, "confidence": 0, "summary": ""}
    return data


def read_list(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        return []


# ---------------------- MAIN ----------------------
rss_sources = read_list(RSS_FILE)
hospitals = read_list(HOSPITAL_FILE)
print(f"Loaded {len(rss_sources)} RSS feeds and {len(hospitals)} hospital URLs")

rows = []
for url in rss_sources:
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:30]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()
            text = f"{title}\n{summary}"
            ai = summarize_text(text)
            if ai["relevance"] == 1 and ai["confidence"] >= CONF_THRESHOLD:
                rows.append({
                    "date_detected": time.strftime("%Y-%m-%d"),
                    "isotope": "Lu-177/Tb-161",
                    "grade": "unspecified",
                    "procurement_type": "tender",
                    "country": "",
                    "issuer": "",
                    "publication_date": "",
                    "deadline_date": "",
                    "estimated_value": "",
                    "confidence": ai["confidence"],
                    "title": title,
                    "summary": ai["summary"],
                    "url": link,
                    "source_type": "RSS"
                })
    except Exception as e:
        print("Feed error:", url, e)

# simple scrape from hospital procurement pages
for site in hospitals:
    try:
        html = requests.get(site, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        ai = summarize_text(text[:3000])
        if ai["relevance"] == 1 and ai["confidence"] >= CONF_THRESHOLD:
            rows.append({
                "date_detected": time.strftime("%Y-%m-%d"),
                "isotope": "Lu-177/Tb-161",
                "grade": "unspecified",
                "procurement_type": "tender",
                "country": "",
                "issuer": site.split("//")[-1].split("/")[0],
                "publication_date": "",
                "deadline_date": "",
                "estimated_value": "",
                "confidence": ai["confidence"],
                "title": f"Hospital Tender — {site}",
                "summary": ai["summary"],
                "url": site,
                "source_type": "Hospital"
            })
    except Exception as e:
        print("Hospital error:", site, e)

print(f"Detected {len(rows)} potential tenders")

# ---------------------- WRITE OUTPUT ----------------------
if rows:
    df = pd.DataFrame(rows)
    # write CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"Stored {len(df)} new records → {OUT_CSV}")

    # write to Google Sheet
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB, rows=1000, cols=20)

    headers = ["date_detected","isotope","grade","procurement_type","country","issuer",
               "publication_date","deadline_date","estimated_value","confidence",
               "title","summary","url","source_type"]
    first_row = ws.row_values(1)
    if first_row != headers:
        ws.resize(rows=1, cols=len(headers))
        ws.update('A1', [headers])

    ws.append_rows(df.values.tolist(), value_input_option="RAW")
    print(f"Appended {len(df)} rows to Google Sheet.")
else:
    print("No relevant tenders found.")
