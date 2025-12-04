PitchSpark ⚡ - AI Profile & Resume Analyzer

PitchSpark is an interactive web app built in Python and Streamlit that acts as a personal AI Career Coach. It uses a Dual-Analysis Engine to provide instant, actionable feedback on LinkedIn profiles and resumes, helping users get discovered and land their next role.



🚀 Core Features

This app doesn't just find problems—it fixes them.

Gamified Score Metric: An instant "Overall Score" (0-100) in the sidebar that provides a tangible grade and motivates users to improve.

Dual-Analysis Engine: Combines a strategic AI "Coach" (GitHub Models API) for high-level, human-like feedback with a programmatic NLP "Assistant" (spacy) for finding specific, rule-based errors (like weak verbs).

One-Click AI Rewrites: A high-impact "Rewrite" button that instantly generates three new, professionally written versions of the user's text in different styles (e.g., "Concise," "Story-Driven").

Flexible Input Tabs: A clean, tabbed UI that seamlessly handles both pasted text (for LinkedIn) and direct PDF resume uploads (using PyPDF2).

🛠️ Technology Stack

This project was built entirely in Python using the following key libraries:

Web Framework: Streamlit

AI Model: GitHub Models API (via requests)

NLP Analysis: spacy

PDF Parsing: PyPDF2

Secrets Management: python-dotenv

🏃 How to Run Locally

You can run this project on your local machine in just a few steps.

1. Prerequisites

Python 3.9+

A GitHub Account (to create a Personal Access Token)

2. Setup Instructions

1. Clone the repository:

git clone 
[https://github.com/devsoni1214/pitchspark-analyzer.git]
(https://github.com/devsoni1214/pitchspark-analyzer.git)
cd pitchspark-analyzer


2. Create and activate a virtual environment:

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


3. Install the required libraries:


pip install -r requirements.txt


4. Download the spacy language model:

python -m spacy download en_core_web_sm


5. Create your secret API key file:
This is the most important step for connecting to the AI.

Create a new file in the project folder named .env

Go to your GitHub PAT settings and generate a new Fine-grained token.

Give it Read-only access to the "Models" permission (under "Account Permissions").

Copy the token (it will start with github_pat_...) and paste it into your .env file like this:

GITHUB_PAT=github_pat_...YOUR_NEW_TOKEN_HERE


3. Run the App

You're all set! Run the following command in your terminal:

streamlit run app.py


Your browser will automatically open, and your AI Career Coach will be ready to use!