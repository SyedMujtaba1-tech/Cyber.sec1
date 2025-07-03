import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ========================
# 1. ENHANCED DATA LOADING
# ========================
print("\n=== Phishing Email Detector ===")

try:
    data = pd.read_csv("emails.csv")
    print("✔ Data loaded successfully")
    
    # === DATA VALIDATION ===
    if not all(col in data.columns for col in ['text', 'label']):
        raise ValueError("CSV must contain 'text' and 'label' columns")
    if not all(label in ['phishing', 'legitimate'] for label in data['label'].unique()):
        raise ValueError("Labels must be 'phishing' or 'legitimate'")
    if len(data) < 10:
        print("⚠ Warning: For better accuracy, add at least 10 emails (5 phishing, 5 legitimate)")
    
    print(f"Total emails: {len(data)} (Phishing: {sum(data['label']=='phishing')}, Legitimate: {sum(data['label']=='legitimate')})")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# ========================
# 2. IMPROVED MODEL TRAINING (FIXED)
# ========================
X = data["text"]
y = data["label"]

# Fixed: Added stratify=y and proper parenthesis
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Ensures balanced samples
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1, 2)  # Fixed: Added missing parenthesis
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train_vec, y_train)

# ========================
# 3. DATABASE INTEGRATION
# ========================
conn = sqlite3.connect('phishing_detector.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS detections
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
               email_text TEXT NOT NULL,
               prediction TEXT NOT NULL,
               confidence REAL,
               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS false_positives
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
               email_text TEXT NOT NULL,
               actual_label TEXT NOT NULL,
               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# ========================
# 4. ENHANCED PREDICTION SYSTEM
# ========================
def analyze_email(email):
    """Process email with validation"""
    if not isinstance(email, str) or not email.strip():
        raise ValueError("Empty email text")
    if len(email) > 1000:
        raise ValueError("Email too long (max 1000 chars)")
        
    vec = vectorizer.transform([email])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0].max()
    return prediction, round(proba * 100, 2)

def save_to_db(email, prediction, confidence):
    """Save with validation"""
    if not email or not prediction:
        print("⚠ Skipping save: Empty data")
        return
    try:
        cursor.execute("INSERT INTO detections (email_text, prediction, confidence) VALUES (?, ?, ?)",
                     (email, prediction, confidence))
        conn.commit()
    except sqlite3.Error as e:
        print(f"⚠ Database error: {e}")

# ========================
# 5. PERFORMANCE EVALUATION (IMPROVED)
# ========================
print("\n=== Model Evaluation ===")
if len(X_test) > 0:  # Only evaluate if test samples exist
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, zero_division=0))
else:
    print("⚠ Evaluation skipped: Not enough test samples")

# ========================
# 6. INTERACTIVE INTERFACE
# ========================
print("\n=== Interactive Mode ===")
print("Type 'quit' to exit | 'report' to show metrics\n")

while True:
    email = input("Enter email text: ").strip()
    
    if email.lower() == 'quit':
        break
        
    if email.lower() == 'report':
        print("\nDatabase Summary:")
        print(f"Total detections: {cursor.execute('SELECT COUNT(*) FROM detections').fetchone()[0]}")
        print(f"False positives: {cursor.execute('SELECT COUNT(*) FROM false_positives').fetchone()[0]}")
        continue
    
    # === INPUT VALIDATION ===
    if not email:
        print("❗ Error: Empty input")
        continue
    if len(email) < 10:
        print("❗ Error: Email too short (min 10 chars)")
        continue
    
    try:
        prediction, confidence = analyze_email(email)
        print(f"Result: {prediction.upper()} ({confidence}% confidence)")
        
        save_to_db(email, prediction, confidence)
        
        feedback = input("Was this correct? (y/n): ").lower()
        if feedback == 'n':
            actual_label = input("What should it be? (phishing/legitimate): ").strip().lower()
            if actual_label in ['phishing', 'legitimate']:
                cursor.execute("INSERT INTO false_positives (email_text, actual_label) VALUES (?, ?)",
                             (email, actual_label))
                conn.commit()
                print("✓ Feedback saved")
            else:
                print("❗ Invalid label (use 'phishing' or 'legitimate')")
                
    except Exception as e:
        print(f"❌ Error: {e}")

# ========================
# 7. CLEANUP
# ========================
conn.close()
print("\n=== Session Ended ===")
print("All results saved to phishing_detector.db")