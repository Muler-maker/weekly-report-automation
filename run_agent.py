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

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RSS_FILE = os.getenv("RSS_FILE", "feeds_rss.txt")
HOSPITAL_FILE = os.getenv("HOSPITAL_FILE", "hospitals.txt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
TAB = os.getenv("GOOGLE_SHEET_TAB", "Sheet1")

SA_JSON = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(SA_JSON, scopes=scopes)
gc = gspread.authorize(creds)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(text: str):
    prompt = f"""
You are a research assistant. Read the following tender excerpt.
Decide if it is relevant to the supply or procurement of Lutetium-177 or Terbium-161.
Respond as JSON:
{{"relevance":1 or 0,"confidence":0-1,"summary":"short English summary"}}
Text:
{text}
"""
    try:
        resp = client.responses.create(model=MODEL, input=prompt, max_output_tokens=250)
        data = json.loads(resp.output_text)
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        data = {"relevance": 0, "confidence": 0, "summary": ""}
    return data

def read_list(file):
    try:
        with open(file, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        return []

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
                rows.append([
                    time.strftime("%Y-%m-%d"),
                    "Lu-177/Tb-161",
                    "tender",
                    title,
                    ai["summary"],
                    link,
                    ai["confidence"],
                    "RSS"
                ])
    except Exception as e:
        print("Feed error:", url, e)

for site in hospitals:
    try:
        html = requests.get(site, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        ai = summarize_text(text[:3000])
        if ai["relevance"] == 1 and ai["confidence"] >= CONF_THRESHOLD:
            rows.append([
                time.strftime("%Y-%m-%d"),
                "Lu-177/Tb-161",
                "tender",
                f"Hospital Tender — {site}",
                ai["summary"],
                site,
                ai["confidence"],
                "Hospital"
            ])
    except Exception as e:
        print("Hospital error:", site, e)

print(f"Detected {len(rows)} potential tenders")

if rows:
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB, rows=1000, cols=10)

    headers = ["date_detected","isotope","type","title","summary","url","confidence","source"]
    if ws.row_values(1) != headers:
        ws.resize(rows=1, cols=len(headers))
        ws.update('A1', [headers])

    ws.append_rows(rows, value_input_option="RAW")
    print(f"✅ Appended {len(rows)} rows to Google Sheet.")
else:
    print("ℹ️ No relevant tenders found.")
