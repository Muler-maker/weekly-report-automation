import os, time, json, re
import feedparser, requests
from bs4 import BeautifulSoup
from google.oauth2.service_account import Credentials
import gspread
from openai import OpenAI

# ---------- Config from env ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RSS_FILE = os.getenv("RSS_FILE", "feeds_rss.txt")
HOSPITAL_FILE = os.getenv("HOSPITAL_FILE", "hospitals.txt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
TAB = os.getenv("GOOGLE_SHEET_TAB", "Sheet1")
DEBUG = os.getenv("DEBUG", "0") == "1"
MATCH_ALL = os.getenv("MATCH_ALL", "0") == "1"   # if True, skip keyword prefilter (for a smoke test)

# ---------- Keyword variants (prefilter before GPT) ----------
KW = [
    r"\blu[\s\-]?177\b", r"\blu177\b", r"\b177\s*lu\b", r"\b177lu\b", r"\b177-lu\b",
    r"\blutetium\b", r"\blutezio\b", r"\blutetio\b", r"\blu-?177\b",
    r"\btb[\s\-]?161\b", r"\btb161\b", r"\b161\s*tb\b", r"\b161tb\b", r"\b161-tb\b",
    r"\bterbium\b", r"\bterbio\b"
]
KW_RE = re.compile("|".join(KW), re.IGNORECASE)

# ---------- Google auth + OpenAI ----------
SA_JSON = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
creds = Credentials.from_service_account_info(
    SA_JSON, scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
gc = gspread.authorize(creds)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Helpers ----------
def summarize_text(text: str):
    """Ask GPT to confirm Lu-177/Tb-161 relevance."""
    prompt = f"""You classify procurement items. Is this about supply/procurement of Lutetium-177 (Lu-177, 177Lu) or Terbium-161 (Tb-161, 161Tb)?
Return JSON only:
{{"relevance":0 or 1, "confidence": number 0..1, "summary":"<=40 words English"}}

Text:
{text}"""
    try:
        r = client.responses.create(model=MODEL, input=prompt, max_output_tokens=200)
        data = json.loads(r.output_text)
        if not isinstance(data, dict):
            raise ValueError("non-dict")
        return data
    except Exception as e:
        if DEBUG: print("GPT parse fail:", e)
        return {"relevance": 0, "confidence": 0, "summary": ""}

def read_list(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    except FileNotFoundError:
        return []

def ensure_sheet(ws):
    headers = ["date_detected","isotope","type","title","summary","url","confidence","source"]
    if ws.row_values(1) != headers:
        ws.resize(rows=1, cols=len(headers))
        # gspread 6.x prefers values then range via named kwargs; this is fine:
        ws.update(values=[headers], range_name="A1")

def get_existing_urls(ws):
    try:
        col = ws.col_values(6)  # URL column
        return set(u.strip() for u in col[1:] if u.strip())
    except Exception:
        return set()

def append_rows(ws, rows):
    if rows:
        ws.append_rows(rows, value_input_option="RAW")
        print(f"✅ Appended {len(rows)} rows to Google Sheet.")

# ---------- Load sources ----------
rss_sources = read_list(RSS_FILE)
hospitals = read_list(HOSPITAL_FILE)
print(f"Loaded {len(rss_sources)} RSS feeds and {len(hospitals)} hospital URLs")

rows = []
total_entries = 0

# ---------- Open Sheet + dedupe set ----------
sh = gc.open_by_key(SHEET_ID)
try:
    ws = sh.worksheet(TAB)
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(title=TAB, rows=1000, cols=10)
ensure_sheet(ws)
existing = get_existing_urls(ws)

# ---------- Process RSS feeds ----------
for url in rss_sources:
    try:
        feed = feedparser.parse(url)
        entries = feed.entries or []
        total_entries += len(entries)
        if DEBUG: print(f"Feed {url} → {len(entries)} entries")
        for e in entries[:100]:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            summary = BeautifulSoup(e.get("summary", "") or "", "html.parser").get_text()
            blob = f"{title}\n{summary}"
            if not MATCH_ALL and not KW_RE.search(blob):
                continue  # fast skip
            if link and link in existing:
                continue  # dedupe by URL
            ai = summarize_text(blob)
            if ai["relevance"] == 1 and ai["confidence"] >= CONF_THRESHOLD:
                rows.append([
                    time.strftime("%Y-%m-%d"),
                    "Lu-177/Tb-161",
                    "tender",
                    title[:240] or "(no title)",
                    ai["summary"],
                    link or url,   # fallback to feed URL if no link
                    round(float(ai["confidence"]), 3),
                    "RSS"
                ])
    except Exception as e:
        if DEBUG: print("Feed error:", url, e)

# ---------- Process hospital procurement pages ----------
for site in hospitals:
    try:
        resp = requests.get(site, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
        status = resp.status_code
        text = BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True)
        if DEBUG: print(f"Hospital {site} → HTTP {status}, chars={len(text)}")
        if status != 200 or len(text) < 500:
            continue
        if not MATCH_ALL and not KW_RE.search(text):
            continue
        if site in existing:
            continue
        ai = summarize_text(text[:3500])
        if ai["relevance"] == 1 and ai["confidence"] >= CONF_THRESHOLD:
            rows.append([
                time.strftime("%Y-%m-%d"),
                "Lu-177/Tb-161",
                "tender",
                f"Hospital Tender — {site}",
                ai["summary"],
                site,
                round(float(ai["confidence"]), 3),
                "Hospital"
            ])
    except Exception as e:
        if DEBUG: print("Hospital error:", site, e)

print(f"Scanned {total_entries} entries across feeds.")
print(f"Detected {len(rows)} potential tenders")

# ---------- Write to Google Sheet ----------
append_rows(ws, rows)

# Helpful debug when no hits
if not rows and DEBUG:
    print("No hits. Sample recent titles per feed (first 3 each):")
    for url in rss_sources:
        try:
            f = feedparser.parse(url)
            print("•", url)
            for e in (f.entries or [])[:3]:
                print("   -", (e.get("title","") or "")[:120])
        except Exception:
            pass
