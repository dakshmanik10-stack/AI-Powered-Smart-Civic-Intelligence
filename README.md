Project Name: AI-Powered Civic Complaint Intelligence System.
Tech Stack: Python, Streamlit, Scikit-Learn, Sentence-Transformers (NLP).

How to run: 
python -m pip install -r requirements.txt 
python -m streamlit run app.py
cd "C:\Users\Shashi\OneDrive\Desktop\Civic Complaint"
.\venv\Scripts\Activate.ps1
streamlit run app.py

Login:
- Admin username: `admin`
- Admin demo password: `admin123`
- Citizens can create an account from the **Citizen Sign up** tab.
- Citizen names must contain letters and spaces only. Passwords must be at
  least 4 characters and include at least one letter and one number.
- Admin navigation has separate **Recent Complaints** and **Admin GIS Dashboard**
  sections. Status changes are made from **Recent Complaints**.

Complaints and status updates are stored locally in `civic_complaints.db` and
remain available after reloading or restarting the app. Citizen accounts and
login sessions are still demo-only and are not stored persistently. The admin
password and demo credentials are not secure; use proper authentication and
database access controls before deploying publicly.