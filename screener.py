"""Resume screening: extract skills, score vs job description (TF-IDF cosine), rank."""
import re, sys, csv
from pathlib import Path
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
SKILLS = ["python","java","sql","machine learning","deep learning","nlp","tensorflow","pytorch",
          "scikit-learn","pandas","numpy","flask","django","fastapi","react","javascript","html","css",
          "aws","docker","git","data analysis","statistics","power bi","tableau","excel","mongodb","mysql"]
HEADS = ["education", "experience", "work experience", "skills", "projects"]

def read_resume(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    return path.read_text(errors="ignore")

def extract_skills(text):
    t = text.lower()
    return [s for s in SKILLS if re.search(r"(?<![a-z])" + re.escape(s) + r"(?![a-z])", t)]

def extract_section(text, names):
    """Return text under a heading in `names` until the next known heading."""
    pat = lambda ns: r"(?im)^\s*(?:%s)\s*:?\s*$" % "|".join(ns)
    m = re.search(pat(names), text)
    if not m: return ""
    rest = text[m.end():]
    n = re.search(pat(HEADS), rest)
    return (rest[:n.start()] if n else rest).strip().replace("\n", " ")[:300]

def screen(job_desc, folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".pdf", ".txt"))
    if not files: sys.exit("No resumes found in " + str(folder))
    texts = [read_resume(f) for f in files]
    m = TfidfVectorizer(stop_words="english").fit_transform([job_desc] + texts)
    scores = cosine_similarity(m[0:1], m[1:])[0]
    jd_skills = set(extract_skills(job_desc))
    rows = []
    for f, t, s in zip(files, texts, scores):
        sk = extract_skills(t)
        rows.append({"candidate": f.stem, "score": round(float(s) * 100, 1),
                     "skills": ", ".join(sk),
                     "matched_skills": ", ".join(x for x in sk if x in jd_skills),
                     "education": extract_section(t, ["education"]),
                     "experience": extract_section(t, ["experience", "work experience"])})
    return sorted(rows, key=lambda r: r["score"], reverse=True)

if __name__ == "__main__":
    jd_file = sys.argv[1] if len(sys.argv) > 1 else BASE / "job_description.txt"
    rows = screen(Path(jd_file).read_text(), BASE / "resumes")
    print(f"{'Rank':<5}{'Candidate':<14}{'Score%':<8}Matched skills")
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['candidate']:<14}{r['score']:<8}{r['matched_skills']}")
    with open(BASE / "ranking_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("\nReport saved to ranking_report.csv")
