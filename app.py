import streamlit as st
import os
import requests
from dotenv import load_dotenv
import PyPDF2
import io
import spacy
from spacy.matcher import Matcher
import re
from typing import List, Tuple, Dict, Any

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="PitchSpark 🚀",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------
# Env + GitHub Models Setup
# ---------------------------------------------------
load_dotenv()
github_pat = os.getenv("GITHUB_PAT")

# 2) If not found, *safely* try Streamlit secrets (for Streamlit Cloud)
if not github_pat:
    try:
        github_pat = st.secrets["GITHUB_PAT"]
    except Exception:
        github_pat = None  # no secrets.toml or no key, ignore

if not github_pat:
    st.error("Error: GITHUB_PAT not found. Please add it in Streamlit Secrets.")
    st.stop()

API_URL = "https://models.github.ai/inference/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"
HEADERS = {
    "Authorization": f"Bearer {github_pat}",
    "Content-Type": "application/json"
}

# ---------------------------------------------------
# NLP / spaCy Setup
# ---------------------------------------------------
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        st.error("Spacy model `en_core_web_sm` not found. Please ensure it is installed.")
        st.stop()

nlp = load_spacy_model()

# ---------------------------------------------------
# Helper: AI Call
# ---------------------------------------------------
def call_github_chat(
    user_content: str,
    system_content: str = "You are PitchSpark, an AI career and resume assistant."
) -> str:
    """Call GitHub Models Chat API and return the assistant content."""
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
    except Exception as e:
        st.error(f"Unexpected error calling GitHub AI: {e}")
    return ""

# ---------------------------------------------------
# Helper: PDF Text Extraction
# ---------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file (bytes)."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# ---------------------------------------------------
# Helper: spaCy Weak Phrase Suggestions
# ---------------------------------------------------
def get_spacy_suggestions(text: str) -> List[str]:
    """Analyze text with Spacy for weak phrases."""
    text = (text or "").strip()
    if not text:
        return []

    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    # Existing weak phrases + some extra soft/filler phrases
    patterns = [
        [{'LOWER': 'i'}, {'LOWER': 'think'}],
        [{'LOWER': 'i'}, {'LOWER': 'believe'}],
        [{'LOWER': 'helped'}, {'LEMMA': 'with'}],
        [{'LOWER': 'responsible'}, {'LOWER': 'for'}],
        [{'LOWER': 'kind'}, {'LOWER': 'of'}],
        [{'LOWER': 'sort'}, {'LOWER': 'of'}],
        [{'LOWER': 'trying'}, {'LOWER': 'to'}],
        [{'LOWER': 'very'}, {'LOWER': 'passionate'}],
    ]

    matcher.add('WEAK_PHRASE', patterns)
    matches = matcher(doc)

    suggestions: List[str] = []
    for match_id, start, end in matches:
        span = doc[start:end]
        text_span = span.text.lower()
        suggestion = ""

        if text_span in ("i think", "i believe"):
            suggestion = f"Found: **'{span.text}'** → Try a more confident phrase."

        elif text_span.startswith("helped"):
            suggestion = (
                f"Found: **'{span.text}'** → Use a stronger verb like "
                "'Assisted', 'Supported', or 'Contributed to'."
            )

        elif text_span == "responsible for":
            suggestion = (
                f"Found: **'{span.text}'** → Try verbs like 'Managed', 'Owned', or 'Led'."
            )

        elif text_span in ("kind of", "sort of"):
            suggestion = (
                f"Found: **'{span.text}'** → Remove softening language to sound more decisive."
            )

        elif text_span == "trying to":
            suggestion = (
                f"Found: **'{span.text}'** → Focus on what you **do** rather than what you're trying to do."
            )

        elif text_span == "very passionate":
            suggestion = (
                f"Found: **'{span.text}'** → Show passion with concrete achievements instead of saying it."
            )

        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)

    return suggestions

# ---------------------------------------------------
# Helper: Keyword Parsing
# ---------------------------------------------------
def parse_keywords(raw: str) -> List[str]:
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]

# ---------------------------------------------------
# Scoring Helpers
# ---------------------------------------------------
def calculate_keyword_match_score(
    text: str,
    keywords: List[str]
) -> Tuple[int, List[str], List[str]]:
    """
    Returns keyword match score (0-40), list of matched keywords, list of missing keywords.
    Uses word-boundary checks to avoid accidental substring matches.
    """
    if not keywords:
        return 0, [], []

    text = text or ""
    text_lower = text.lower()
    matched = []
    missing = []

    for kw in keywords:
        kw_lower = kw.lower()
        # Match whole words / phrases, not substrings inside bigger words
        pattern = r"\b" + re.escape(kw_lower) + r"\b"
        if re.search(pattern, text_lower):
            matched.append(kw)
        else:
            missing.append(kw)

    coverage = len(matched) / len(keywords)
    score = int(round(coverage * 40))  # max 40 points

    return score, matched, missing


def calculate_ats_score(text: str, mode: str = "resume") -> Tuple[int, Dict[str, bool]]:
    """
    Simple ATS-friendliness heuristic (0-30).
    Checks for presence of standard section headings and plain-text friendliness.
    Uses heading-like lines and section synonyms.
    """
    text = (text or "").strip()
    if not text:
        return 0, {}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines_lower = [ln.lower() for ln in lines]

    # Section groups with synonyms
    if mode == "resume":
        section_groups: Dict[str, List[str]] = {
            "experience": [
                "experience", "work experience", "professional experience",
                "employment history", "work history"
            ],
            "education": [
                "education", "academic background", "academic qualifications"
            ],
            "projects": [
                "projects", "personal projects", "academic projects"
            ],
            "skills": [
                "skills", "technical skills", "core skills", "key skills"
            ],
            "summary": [
                "summary", "professional summary", "profile", "about me"
            ],
            "objective": [
                "objective", "career objective"
            ],
            "certifications": [
                "certifications", "licenses", "licenses & certifications"
            ],
            "achievements": [
                "achievements", "awards", "honors", "accomplishments"
            ],
            "publications": [
                "publications", "research publications"
            ],
        }
    else:
        # LinkedIn-style sections
        section_groups = {
            "about": ["about", "about me", "summary"],
            "experience": ["experience", "work experience", "professional experience"],
            "projects": ["projects", "featured projects"],
            "skills": ["skills", "top skills"],
            "certifications": ["certifications", "licenses & certifications"],
        }

    section_presence: Dict[str, bool] = {}
    section_hits = 0

    for canonical, variants in section_groups.items():
        present = False
        for ln in lines_lower:
            # Treat short lines as likely headings, but still allow full-text search
            if len(ln.split()) <= 8:
                if any(re.search(r"\b" + re.escape(v) + r"\b", ln) for v in variants):
                    present = True
                    break
        # Fallback: if still not found, check whole text
        if not present:
            for v in variants:
                if re.search(r"\b" + re.escape(v) + r"\b", text.lower()):
                    present = True
                    break

        section_presence[canonical] = present
        if present:
            section_hits += 1

    sections_count = len(section_groups) if section_groups else 0
    coverage_ratio = section_hits / sections_count if sections_count else 0
    section_score = int(round(coverage_ratio * 18))  # 0–18

    # Plain text / formatting checks (very approximate)
    bullet_lines = [ln for ln in lines if ln.startswith(("-", "•", "*"))]
    caps_lines = [ln for ln in lines if len(ln) > 4 and ln.upper() == ln]

    # Bullets: good to have some, not 0
    if len(bullet_lines) == 0:
        bullet_score = 4
    elif 1 <= len(bullet_lines) <= 40:
        bullet_score = 8
    else:
        bullet_score = 6  # too many may be noisy

    # ALL CAPS headings – small positive signal
    if 0 < len(caps_lines) <= 25:
        caps_score = 4
    else:
        caps_score = 2

    ats_score = section_score + bullet_score + caps_score  # max ~30
    ats_score = max(0, min(30, ats_score))

    return ats_score, section_presence


def calculate_language_score(
    text: str,
    spacy_suggestions: List[str],
    mode: str = "resume"
) -> int:
    """
    Basic language & structure score (0–20).
    Penalizes weak phrases, rewards multiple sentences & reasonable length.
    Length targets are different for LinkedIn vs resume.
    """
    text = (text or "").strip()
    if not text:
        return 0

    length = len(text)

    # Length scoring tuned by mode
    if mode == "linkedin":
        # LinkedIn About: usually a solid paragraph / short story
        if length < 150:
            length_score = 4
        elif 150 <= length <= 400:
            length_score = 8
        elif 400 < length <= 1500:
            length_score = 12
        elif 1500 < length <= 3000:
            length_score = 10
        else:
            length_score = 6
    else:
        # Resume: whole document text, can be longer
        if length < 500:
            length_score = 6
        elif 500 <= length <= 2500:
            length_score = 12
        elif 2500 < length <= 6000:
            length_score = 10
        else:
            length_score = 8

    # Sentence structure
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)
    if sentence_count <= 1:
        struct_score = 3
    elif 2 <= sentence_count <= 10:
        struct_score = 8
    elif 11 <= sentence_count <= 25:
        struct_score = 6
    else:
        struct_score = 4

    # Weak phrases penalty
    weak_penalty = min(len(spacy_suggestions) * 3, 8)

    lang_score = length_score + struct_score - weak_penalty
    lang_score = max(0, min(20, lang_score))
    return lang_score


def calculate_impact_score(text: str) -> int:
    """
    Impact score (0–10) based on presence of numbers & action verbs,
    with extra weight when they appear in the same sentence.
    """
    text = (text or "").strip()
    if not text:
        return 0

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    text_lower = text.lower()

    # Action verbs
    action_verbs = [
        "led", "managed", "built", "created", "designed", "developed",
        "launched", "improved", "increased", "reduced", "achieved",
        "optimized", "implemented", "delivered", "owned"
    ]

    # Helpers for metrics
    def has_metric(s: str) -> bool:
        return bool(re.search(r"\d", s) or re.search(r"[%₹$€]", s))

    total_action_hits = sum(1 for v in action_verbs if v in text_lower)
    has_any_metric = any(has_metric(s) for s in sentences)

    # Sentences that contain BOTH metrics and action verbs
    impact_sentences = 0
    for s in sentences:
        s_lower = s.lower()
        if has_metric(s) and any(v in s_lower for v in action_verbs):
            impact_sentences += 1

    # Scoring
    score = 0
    if impact_sentences >= 4:
        score = 10
    elif 2 <= impact_sentences < 4:
        score = 8
    elif impact_sentences == 1:
        score = 6
    else:
        # No strong metric+action pairing, but maybe either metrics or verbs exist
        if has_any_metric and total_action_hits >= 1:
            score = 5
        elif has_any_metric or total_action_hits >= 2:
            score = 3
        else:
            score = 1 if total_action_hits or has_any_metric else 0

    return max(0, min(10, score))


def calculate_all_scores(
    text: str,
    keywords: List[str],
    spacy_suggestions: List[str],
    mode: str = "resume"
) -> Dict[str, Any]:
    """
    Returns dict with:
    - keyword_score
    - ats_score
    - language_score
    - impact_score
    - overall_score
    - interview_probability
    - matched_keywords
    - missing_keywords
    - section_presence
    """
    keyword_score, matched, missing = calculate_keyword_match_score(text, keywords)
    ats_score, section_presence = calculate_ats_score(text, mode=mode)
    language_score = calculate_language_score(text, spacy_suggestions, mode=mode)
    impact_score = calculate_impact_score(text)

    overall_score = keyword_score + ats_score + language_score + impact_score  # max ~100
    overall_score = max(0, min(100, overall_score))

    # Interview probability based more on keyword + ATS
    # (Note: heuristic coaching signal, not a real ATS probability.)
    interview_probability = int(
        round(
            0.6 * (keyword_score / 40 * 100 if 40 else 0) +
            0.3 * (ats_score / 30 * 100 if 30 else 0) +
            0.1 * (language_score / 20 * 100 if 20 else 0)
        )
    )
    interview_probability = max(0, min(100, interview_probability))

    return {
        "keyword_score": keyword_score,
        "ats_score": ats_score,
        "language_score": language_score,
        "impact_score": impact_score,
        "overall_score": overall_score,
        "interview_probability": interview_probability,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "section_presence": section_presence,
    }

# ---------------------------------------------------
# Advanced AI Features
# ---------------------------------------------------
def ai_recruiter_eye(text: str, target_role: str, mode: str = "resume") -> str:
    user_prompt = f"""
Act as a recruiter reviewing a candidate for the role: {target_role or "Not specified"}.
You only have **7 seconds** to scan this {mode}.

1. List the **top 3 things** you will notice first (bullets).
2. List the **top 3 red flags** or confusion points.
3. Give a **one-line first impression** in quotes.

Here is the content:
---
{text}
"""
    return call_github_chat(user_prompt)


def ai_skill_truth_detector(text: str) -> str:
    user_prompt = f"""
You are 'PitchSpark', an AI that checks if claimed skills are backed by evidence.

From the content below, extract a **table** with columns:
- Skill
- Proof Found (Yes/No)
- Evidence Snippet (<=15 words)

Only include skills that seem explicitly or implicitly claimed.
Return the result as a Markdown table.

Content:
---
{text}
"""
    return call_github_chat(user_prompt)


def ai_career_growth_suggestions(text: str, target_role: str) -> str:
    user_prompt = f"""
You are an AI career coach.

Given this person's current profile/resume and target role: {target_role or "Not specified"}:

Provide:
1. **Next 2 possible career moves** (job titles).
2. **Key skills to add** in the next 6–12 months.
3. **One realistic learning roadmap** in bullet points.

Content:
---
{text}
"""
    return call_github_chat(user_prompt)


def ai_personality_radar(text: str) -> str:
    user_prompt = f"""
Analyze the tone and language of the following LinkedIn/About/Resume content.

Rate the following traits from 1–5:
- Confidence
- Leadership
- Learning Mindset
- Communication Clarity
- Professionalism

Return a **Markdown table** with columns:
Trait | Score (1–5) | Short Comment.

Content:
---
{text}
"""
    return call_github_chat(user_prompt)


def ai_rewrite_about_section(text: str) -> str:
    user_prompt = f"""
Act as 'PitchSpark', an expert LinkedIn brand copywriter.

Rewrite this LinkedIn 'About' section in **3 styles**:
1. **Concise & Punchy**
2. **Story-Driven**
3. **Keyword-Optimized for recruiters & ATS**

Use clear Markdown headings (##) for each style.
Preserve the factual content but improve clarity and impact.

Original:
---
{text}
"""
    return call_github_chat(user_prompt)


def ai_extract_jd_keywords(jd_text: str) -> List[str]:
    user_prompt = f"""
From the following job description, extract the **15 most important keywords/skills/tools**.
Return them as a **comma-separated list only**, no extra text.

Job Description:
---
{jd_text}
"""
    raw = call_github_chat(user_prompt)
    return parse_keywords(raw)


def ai_jd_vs_resume_commentary(jd_text: str, resume_text: str, matched: List[str], missing: List[str]) -> str:
    user_prompt = f"""
You are PitchSpark, analyzing JD vs Resume fit.

Job Description:
---
{jd_text}

Resume:
---
{resume_text}

Matched keywords:
{', '.join(matched) if matched else 'None'}

Missing but important keywords:
{', '.join(missing) if missing else 'None'}

Provide:
1. A brief **fit summary** (2–3 sentences).
2. Top **3 changes** to improve alignment with the JD.
"""
    return call_github_chat(user_prompt)


def ai_suggest_keywords_from_role_and_text(target_role: str, text: str) -> List[str]:
    """
    Suggest important keywords when the user hasn't provided any,
    based on their target role and current content.
    """
    user_prompt = f"""
You are PitchSpark, helping optimize a profile/resume.

Target role: {target_role or "Not specified"}

From the content below, suggest **10–15 important skills/keywords/tools**
that should be emphasized for this role.

Return them as a **comma-separated list only**, no extra commentary.

Content:
---
{text}
"""
    raw = call_github_chat(user_prompt)
    return parse_keywords(raw)

# ---------------------------------------------------
# Sidebar – Global Inputs & Score Overview
# ---------------------------------------------------
with st.sidebar:
    st.title("PitchSpark ⚡")
    st.subheader("Your AI Career Coach")

    st.markdown(
        "Paste your LinkedIn/About or upload a resume, and PitchSpark will "
        "analyze **keywords, ATS-friendliness, language, and impact**."
    )

    st.divider()
    st.subheader("Target Context")

    target_role = st.text_input(
        "Your Target Role (optional)",
        placeholder="e.g. Data Analyst, Software Engineer, Product Manager"
    )

    raw_keywords = st.text_area(
        "Important Keywords (comma-separated)",
        placeholder="python, sql, data analysis, power bi, machine learning"
    )
    global_keywords = parse_keywords(raw_keywords)

    st.caption(
        "These keywords are used for **scoring**. You can copy them from a JD, "
        "or type skills you want your profile to reflect."
    )
    if not global_keywords:
        st.caption(
            "_No keywords entered – PitchSpark may auto-suggest some based on your target role when analyzing._"
        )

    st.divider()
    st.subheader("Overall Score (last run)")

    if "score_state" not in st.session_state:
        st.session_state.score_state = {
            "overall": 0,
            "interview_prob": 0
        }

    score_placeholder = st.empty()
    score_placeholder.metric(
        "PitchSpark Score",
        f"{st.session_state.score_state['overall']} / 100",
        "Run an analysis to update"
    )

    interview_placeholder = st.empty()
    interview_placeholder.metric(
        "Interview Probability",
        f"{st.session_state.score_state['interview_prob']} %",
        None
    )

    st.markdown(
        """
**Score Components:**
- 🎯 Keyword Match (40%)
- 🧠 ATS Friendliness (30%)
- 📝 Language & Structure (20%)
- 🚀 Impact (10%)
"""
    )
    st.caption(
        "_These are heuristic coaching scores, not an official ATS simulation or guarantee of interviews._"
    )
    st.caption(
        "Privacy: Your text is sent securely to an AI API (GitHub Models) for analysis. "
        "The app itself does not store your content permanently."
    )

# ---------------------------------------------------
# Main Layout
# ---------------------------------------------------
st.title("PitchSpark 🚀 – AI LinkedIn & Resume Super-Reviewer")
st.markdown(
    "Turn your profile and resume into a **hire-magnet** with keyword-aware, "
    "ATS-friendly, recruiter-style analysis."
)

tab1, tab2, tab3 = st.tabs([
    "LinkedIn Profile Analyzer",
    "PDF Resume Analyzer",
    "JD vs Resume Battle Mode ⚔️"
])

# ---------------------------------------------------
# Tab 1 – LinkedIn Profile Analyzer
# ---------------------------------------------------
with tab1:
    st.header("Analyze Your LinkedIn 'About' Section")

    profile_text = st.text_area(
        "Paste your LinkedIn 'About' section here:",
        height=220,
        placeholder="Write or paste your About section…"
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    # ---- Analyze Profile ----
    with col1:
        if st.button("🔍 Analyze My Profile", key="analyze_profile"):
            if profile_text.strip():
                # Decide which keywords to use (user-provided or auto-suggested)
                effective_keywords = list(global_keywords)
                if not effective_keywords and target_role:
                    suggested = ai_suggest_keywords_from_role_and_text(target_role, profile_text)
                    if suggested:
                        effective_keywords = suggested
                        st.info(
                            "Using auto-suggested keywords for scoring: "
                            + ", ".join(suggested)
                        )

                with st.spinner("Analyzing your LinkedIn profile…"):
                    spacy_suggestions = get_spacy_suggestions(profile_text)
                    scores = calculate_all_scores(
                        profile_text,
                        effective_keywords,
                        spacy_suggestions,
                        mode="linkedin"
                    )

                    # Update sidebar metrics
                    st.session_state.score_state["overall"] = scores["overall_score"]
                    st.session_state.score_state["interview_prob"] = scores["interview_probability"]

                    score_placeholder.metric(
                        "PitchSpark Score",
                        f"{scores['overall_score']} / 100",
                        None
                    )
                    interview_placeholder.metric(
                        "Interview Probability",
                        f"{scores['interview_probability']} %",
                        None
                    )

                    st.subheader("📊 Score Breakdown")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("🎯 Keyword Match", f"{scores['keyword_score']} / 40")
                        st.metric("🧠 ATS Friendliness", f"{scores['ats_score']} / 30")
                    with c2:
                        st.metric("📝 Language & Structure", f"{scores['language_score']} / 20")
                        st.metric("🚀 Impact", f"{scores['impact_score']} / 10")

                    st.progress(scores["overall_score"] / 100)

                    with st.expander("Keyword Coverage Details"):
                        st.write("**Matched Keywords:**", ", ".join(scores["matched_keywords"]) or "None")
                        st.write("**Missing Keywords:**", ", ".join(scores["missing_keywords"]) or "None")

                    analysis_prompt = f"""
Act as an expert LinkedIn reviewer ('PitchSpark') for the role: {target_role or "Not specified"}.

Analyze this 'About' section and provide:
1. **Critique** – strengths & weaknesses.
2. **Top 3 Action Items** – specific edits or improvements.
3. **One liner** – how a recruiter would describe this profile.

Content:
---
{profile_text}
"""
                    ai_analysis = call_github_chat(analysis_prompt)

                    if ai_analysis:
                        st.subheader("✨ AI-Powered Analysis")
                        st.markdown(ai_analysis)

                    if spacy_suggestions:
                        st.subheader("💡 Language Improvement Suggestions")
                        for suggestion in spacy_suggestions:
                            st.warning(suggestion)
            else:
                st.error("Please paste your 'About' section first.")

    # ---- Rewrite About ----
    with col2:
        if st.button("✍️ Rewrite My 'About'", key="rewrite_profile"):
            if profile_text.strip():
                with st.spinner("Generating improved versions…"):
                    rewrites = ai_rewrite_about_section(profile_text)
                    if rewrites:
                        st.subheader("AI-Powered Rewrites")
                        st.markdown(rewrites)
            else:
                st.error("Please paste your 'About' section first.")

    # ---- Recruiter Eye Mode ----
    with col3:
        if st.button("👀 Recruiter Eye Mode", key="recruiter_profile"):
            if profile_text.strip():
                with st.spinner("Simulating a recruiter scanning your profile…"):
                    eye = ai_recruiter_eye(profile_text, target_role, mode="profile")
                    if eye:
                        st.subheader("Recruiter Eye View (7-second scan)")
                        st.markdown(eye)
            else:
                st.error("Please paste your 'About' section first.")

    # Extra advanced insights
    st.divider()
    st.subheader("Advanced Insights")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🧪 Skill Truth Detector (Profile)", key="skill_truth_profile"):
            if profile_text.strip():
                with st.spinner("Checking claimed skills vs evidence…"):
                    truth = ai_skill_truth_detector(profile_text)
                    if truth:
                        st.markdown(truth)
            else:
                st.error("Please paste your 'About' section first.")

    with col_b:
        if st.button("🧭 Personality & Soft-Skill Radar", key="personality_profile"):
            if profile_text.strip():
                with st.spinner("Analyzing personality signals…"):
                    radar = ai_personality_radar(profile_text)
                    if radar:
                        st.markdown(radar)
            else:
                st.error("Please paste your 'About' section first.")

    if st.button("📈 Career Growth Suggestions (from Profile)", key="career_profile"):
        if profile_text.strip():
            with st.spinner("Designing your next career moves…"):
                growth = ai_career_growth_suggestions(profile_text, target_role)
                if growth:
                    st.subheader("Career Growth Suggestions")
                    st.markdown(growth)
        else:
            st.error("Please paste your 'About' section first.")

# ---------------------------------------------------
# Tab 2 – PDF Resume Analyzer
# ---------------------------------------------------
with tab2:
    st.header("Analyze Your PDF Resume")

    resume_file = st.file_uploader("Upload your resume (PDF only)", type="pdf", key="resume_file_tab2")

    if resume_file is not None:
        file_bytes = resume_file.read()
        with st.spinner("Reading your resume…"):
            resume_text = extract_text_from_pdf(file_bytes)

        if not resume_text:
            st.error("Could not extract text from this PDF. It might be a scanned image. Try another file or copy-paste the text.")
        else:
            st.success("Resume text extracted successfully.")

            if st.button("🔍 Analyze My Resume", key="analyze_resume"):
                # Decide which keywords to use (user-provided or auto-suggested)
                effective_keywords = list(global_keywords)
                if not effective_keywords and target_role:
                    suggested = ai_suggest_keywords_from_role_and_text(target_role, resume_text)
                    if suggested:
                        effective_keywords = suggested
                        st.info(
                            "Using auto-suggested keywords for scoring: "
                            + ", ".join(suggested)
                        )

                with st.spinner("Analyzing your resume…"):
                    spacy_suggestions = get_spacy_suggestions(resume_text)
                    scores = calculate_all_scores(
                        resume_text,
                        effective_keywords,
                        spacy_suggestions,
                        mode="resume"
                    )

                    # Update sidebar metrics
                    st.session_state.score_state["overall"] = scores["overall_score"]
                    st.session_state.score_state["interview_prob"] = scores["interview_probability"]

                    score_placeholder.metric(
                        "PitchSpark Score",
                        f"{scores['overall_score']} / 100",
                        None
                    )
                    interview_placeholder.metric(
                        "Interview Probability",
                        f"{scores['interview_probability']} %",
                        None
                    )

                    st.subheader("📊 Score Breakdown")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("🎯 Keyword Match", f"{scores['keyword_score']} / 40")
                        st.metric("🧠 ATS Friendliness", f"{scores['ats_score']} / 30")
                    with c2:
                        st.metric("📝 Language & Structure", f"{scores['language_score']} / 20")
                        st.metric("🚀 Impact", f"{scores['impact_score']} / 10")

                    st.progress(scores["overall_score"] / 100)

                    with st.expander("Keyword Coverage Details"):
                        st.write("**Matched Keywords:**", ", ".join(scores["matched_keywords"]) or "None")
                        st.write("**Missing Keywords:**", ", ".join(scores["missing_keywords"]) or "None")

                    with st.expander("ATS Section Detection"):
                        for section, present in scores["section_presence"].items():
                            st.write(f"- `{section}`: {'✅' if present else '❌'}")

                    analysis_prompt = f"""
Act as an expert Resume reviewer ('PitchSpark') for the role: {target_role or "Not specified"}.

Analyze this resume and provide:
1. **Critique** – strengths & weaknesses.
2. **Top 3 Action Items** to improve chances.
3. **ATS Advice** – anything that might confuse parsing.

Resume text:
---
{resume_text}
"""
                    ai_analysis = call_github_chat(analysis_prompt)
                    if ai_analysis:
                        st.subheader("✨ AI-Powered Resume Analysis")
                        st.markdown(ai_analysis)

                    if spacy_suggestions:
                        st.subheader("💡 Language Improvement Suggestions")
                        for suggestion in spacy_suggestions:
                            st.warning(suggestion)

            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                if st.button("👀 Recruiter Eye Mode (Resume)", key="recruiter_resume"):
                    with st.spinner("Simulating a recruiter scanning your resume…"):
                        eye = ai_recruiter_eye(resume_text, target_role, mode="resume")
                        if eye:
                            st.subheader("Recruiter Eye View (7-second scan)")
                            st.markdown(eye)

            with col_r2:
                if st.button("🧪 Skill Truth Detector (Resume)", key="skill_truth_resume"):
                    with st.spinner("Checking skills vs evidence…"):
                        truth = ai_skill_truth_detector(resume_text)
                        if truth:
                            st.subheader("Skill Truth Detector")
                            st.markdown(truth)

            with col_r3:
                if st.button("📈 Career Growth Suggestions (Resume)", key="career_resume"):
                    with st.spinner("Designing your growth roadmap…"):
                        growth = ai_career_growth_suggestions(resume_text, target_role)
                        if growth:
                            st.subheader("Career Growth Suggestions")
                            st.markdown(growth)

# ---------------------------------------------------
# Tab 3 – JD vs Resume Battle Mode
# ---------------------------------------------------
with tab3:
    st.header("JD vs Resume Battle Mode ⚔️")

    jd_text = st.text_area(
        "Paste Job Description (JD):",
        height=200,
        placeholder="Paste the JD here to check how well your resume matches it…"
    )

    jd_resume_file = st.file_uploader(
        "Upload the Resume to compare (PDF only)",
        type="pdf",
        key="resume_file_tab3"
    )

    use_profile_instead = st.checkbox(
        "Use my LinkedIn 'About' text from Tab 1 instead of a resume (optional)",
        value=False
    )

    if st.button("⚔️ Analyze JD vs Resume Match", key="analyze_jd_vs_resume"):
        if not jd_text.strip():
            st.error("Please paste a Job Description first.")
        else:
            # Determine source text
            candidate_text = ""
            label = ""
            if use_profile_instead and profile_text.strip():
                candidate_text = profile_text
                label = "LinkedIn About"
            elif jd_resume_file is not None:
                bytes_file = jd_resume_file.read()
                candidate_text = extract_text_from_pdf(bytes_file)
                label = "Resume"
            else:
                st.error("Please either upload a resume or enable 'Use my LinkedIn About' with text in Tab 1.")
                candidate_text = ""

            if candidate_text:
                with st.spinner("Extracting keywords from JD and matching…"):
                    jd_keywords = ai_extract_jd_keywords(jd_text)
                    if not jd_keywords:
                        st.error("Could not extract keywords from the JD. Try simplifying the JD text.")
                    else:
                        kw_score, matched_kw, missing_kw = calculate_keyword_match_score(
                            candidate_text,
                            jd_keywords
                        )

                        st.subheader("🎯 JD Keyword Coverage")

                        st.metric(
                            "JD Keyword Match Score",
                            f"{kw_score} / 40",
                            None
                        )

                        st.write(f"**Source analyzed:** {label}")

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.write("✅ **Matched JD Keywords:**")
                            if matched_kw:
                                st.write(", ".join(matched_kw))
                            else:
                                st.write("_None_")

                        with col_m2:
                            st.write("❌ **Missing JD Keywords:**")
                            if missing_kw:
                                st.write(", ".join(missing_kw))
                            else:
                                st.write("_None_")

                        commentary = ai_jd_vs_resume_commentary(jd_text, candidate_text, matched_kw, missing_kw)
                        if commentary:
                            st.subheader("📣 Fit Summary & Recommendations")
                            st.markdown(commentary)
