import os
import json
import time
import re
from typing import Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import EXTERNAL_PATH, INTERIM_PATH

#Main url of Dictionary 
CONCEPT_INDEX_URL = (
    "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_eng.html"
)

#Output Json of video info
OUT_PATH = "videos.json"


def norm_sign(s):
    s = s.strip().lower()

    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)

    # Need to add more fixes("Im Shit at spelling : (")
    typo_fixes = {
        "celebbrate": "celebrate",
        "enviroment": "environment",
        "feild": "field",
        "theartre": "theatre",          
        "refridgerator": "refrigerator"
    }
    s = typo_fixes.get(s, s)

    s = re.sub(r"\s*\d+$", "", s)  

    return s
  
  
def fetch_soup(url: str, session: requests.Session):
    r = session.get(url, timeout=30) #Get session
    r.raise_for_status() #get error code if there is one
    r.encoding = "utf-8" #Need uft-8 for HamNoSYs
    return BeautifulSoup(r.text, "html.parser") #Create Beautiful soup object


def build_concept_map(session: requests.Session) -> Dict[str, str]:

    soup = fetch_soup(CONCEPT_INDEX_URL, session) #parce the list of signs

    concept_map = {} 

    #loop through links
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True) #Extract sign name
        href = a["href"] #extract link url

        if not text:
            continue

        #Makes sure its the right link
        if href.endswith(".html") and (href.startswith("cs/") or "/cs/" in href):
            key = norm_sign(text) #Correcting sign names from my dumb spelling
            concept_map[key] = urljoin(CONCEPT_INDEX_URL, href) #makes sure the link isnt broken, website had weird urls
            print("Example concept_map entries:", list(concept_map.items())[:5])

    return concept_map


def extract_bsl_hamnosys(concept_url: str, session: requests.Session) -> Optional[str]:
    
    soup = fetch_soup(concept_url, session)

    #Creates list of text frome page 
    lines = [ln.strip() for ln in soup.get_text("\n").split("\n") if ln.strip()]

    #Check if line has valid stmbols eg. HamNoSys
    def has_pua(s):
        return any("\uE000" <= ch <= "\uF8FF" for ch in s)

    #Find specifically the BSL Box
    try:
        i = lines.index("BSL")
    except ValueError:
        i = -1
    #Once found BSL Box find the text in PUA eg HamNosys
    if i != -1:
        for ln in lines[i + 1 : i + 25]: 
            if has_pua(ln):
                return ln

    #Just in case search page again
    for ln in lines:
        if has_pua(ln):
            return ln

    return None


def make_dataset():
    
    #session
    session = requests.Session()
    
    #I'm a real boy I swear
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (dataset builder; contact: research use)"
        }
    )

    print("Building concept map from Dicta-Sign…")
    concept_map = build_concept_map(session) # My dictionsry of sign to link
    print(f"Found {len(concept_map)} concept links.")

    videos = []
    hand_landmarks = []
    
    filenames = sorted(os.listdir(EXTERNAL_PATH))
    InterimFilesnames = sorted(os.listdir(INTERIM_PATH))

    for idx, filename in enumerate(filenames):
        if not filename.lower().endswith(".mp4"):
            continue

        sign_raw = filename[:-4] 
        
        for id, Jsonfilename in enumerate(InterimFilesnames):
            if Jsonfilename.lower().endswith(".json"):
                
                hand_landmarks.append(Jsonfilename)
            
        sign_key = norm_sign(sign_raw)#Get speeling right

        concept_url = concept_map.get(sign_key)

        #Try fix if concept url didnt work
        if not concept_url:
            stripped = re.sub(r"\d+$", "", sign_raw.lower())
            stripped = norm_sign(stripped)
            concept_url = concept_map.get(stripped)
        hamnosys = ""


        if concept_url:
            try:
                ham = extract_bsl_hamnosys(concept_url, session) # GetHamNoSys
                if ham:
                    hamnosys = ham
                    print(f"[OK] {sign_raw}: {hamnosys[:20]}... (len={len(hamnosys)})")
            except Exception as e:
                print(f"[WARN] Failed to scrape {sign_raw} ({concept_url}): {e}")

            time.sleep(0.2)
        else:
            print(f"[MISS] No Dicta-Sign concept match for: {sign_raw}")

        videos.append(
            {
                "id": len(videos) + 1,
                "Sign": sign_raw,
                "filepath": os.path.join(EXTERNAL_PATH, filename),
                "HamNoSys": hamnosys,
                "concept_url": concept_url or "",
                "HandLandmark": os.path.join(INTERIM_PATH,hand_landmarks[idx])
            }
        )

    #
    data = {"videos": videos}

    #Open Videos JSON and write to it
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {OUT_PATH}")

    filled = sum(1 for v in videos if v["HamNoSys"]) #How many successful
    missing = len(videos) - filled #How many failed

    print(f"HamNoSys filled: {filled}/{len(videos)}")
    print(f"Missing HamNoSys: {missing}")

    #Didnt work
    if missing:
        print("\nUnmatched signs:")
        for v in videos:
            if not v["HamNoSys"]:
                print("-", v["Sign"])
        print(f"HamNoSys filled: {filled}/{len(videos)}")

#Run Program 
if __name__ == "__main__":
    make_dataset()