import re
import time
from typing import Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

#The Dict-Sign Website 
CONCEPT_INDEX = {
    "bsl": "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_eng.html",
    "dgs": "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_ger.html",
    "lsf": "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_fre.html",
    "gsl": "https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_gre.html",
}


#Some typos in english sign names, will fix later
def norm_sign(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)

    typo_fixes = {
        "celebbrate": "celebrate",
        "enviroment": "environment",
        "feild": "field",
        "theartre": "theatre",
        "refridgerator": "refrigerator",
    }
    s = typo_fixes.get(s, s)

    s = re.sub(r"\s*\d+$", "", s)
    return s

#Web Scraper
def fetch_soup(url: str, session: requests.Session) -> BeautifulSoup:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return BeautifulSoup(r.text, "html.parser")


def build_concept_map_for_language(lang_code: str, session: requests.Session) -> Dict[str, str]:
    """
    lang_code: "BSL", "DGS", "LSF", "GSL"
    Returns: {normalized_gloss: concept_url}
    """
    index_url = CONCEPT_INDEX[lang_code.lower()]
    soup = fetch_soup(index_url, session)

    concept_map: Dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]

        if not text:
            continue

        if href.endswith(".html") and ("cs/" in href):
            key = norm_sign(text)
            concept_map[key] = urljoin(index_url, href)

    return concept_map


def extract_hamnosys_for_language(
    concept_url: str, session: requests.Session, lang_code: str
) -> Optional[str]:
    """
    lang_code must be one of: "BSL", "DGS", "GSL", "LSF"
    """
    soup = fetch_soup(concept_url, session)
    lines = [ln.strip() for ln in soup.get_text("\n").split("\n") if ln.strip()]

    def has_pua(s: str) -> bool:
        return any("\uE000" <= ch <= "\uF8FF" for ch in s)

    try:
        i = lines.index(lang_code)
    except ValueError:
        i = -1

    if i != -1:
        for ln in lines[i + 1 : i + 25]:
            if has_pua(ln):
                return ln

    for ln in lines:
        if has_pua(ln):
            return ln

    return None


def load_all_concept_maps(session: requests.Session) -> Dict[str, Dict[str, str]]:
 
    maps: Dict[str, Dict[str, str]] = {}
    for lang in ["BSL", "DGS", "LSF", "GSL"]:
        print(f"[concept_maps] Building concept map for {lang}...")
        maps[lang] = build_concept_map_for_language(lang, session)
        time.sleep(0.5)
    return maps
