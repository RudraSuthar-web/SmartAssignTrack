from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import os
import PyPDF2
import pytesseract
from PIL import Image
import pdf2image
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
from datetime import datetime
import uvicorn
import shutil
import logging
import re
import hashlib
from fuzzywuzzy import fuzz

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_NAME = "assignments.db"
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE,
                      password TEXT,
                      role TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS assignments
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      title TEXT,
                      description TEXT,
                      deadline TEXT,
                      professor_id INTEGER,
                      keywords TEXT,
                      FOREIGN KEY (professor_id) REFERENCES users(id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS submissions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      assignment_id INTEGER,
                      student_id INTEGER,
                      file_path TEXT,
                      file_hash TEXT,
                      submission_time TEXT,
                      grade REAL,
                      plagiarism_score REAL,
                      quality_score REAL,
                      FOREIGN KEY (assignment_id) REFERENCES assignments(id),
                      FOREIGN KEY (student_id) REFERENCES users(id))''')
        conn.commit()

init_db()

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class AssignmentCreate(BaseModel):
    title: str
    description: str
    deadline: str
    keywords: Optional[str]

class SubmissionResponse(BaseModel):
    id: int
    assignment_id: int
    student_id: int
    file_path: str
    submission_time: str
    grade: Optional[float]
    plagiarism_score: Optional[float]
    quality_score: Optional[float]

# Enhanced ML Model for grading and plagiarism detection
class AssignmentChecker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.academic_terms = [
            'equation', 'function', 'theory', 'analysis', 'system', 'model',
            'algorithm', 'data', 'hypothesis', 'experiment', 'proof', 'theorem'
        ]

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            images = pdf2image.convert_from_path(file_path)
            text = ""
            for image in images:
                image = image.convert('L')  # Convert to grayscale
                image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Binarize
                text += pytesseract.image_to_string(image, lang='eng') + "\n"
            text = re.sub(r'\s+', ' ', text.strip())
            if not text:
                logger.warning(f"No text extracted from PDF: {file_path}")
            else:
                logger.info(f"Extracted text from {file_path}: {text[:100]}...")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""

    def check_plagiarism(self, submission_text: str, other_submissions: List[str], current_file_path: str) -> float:
        try:
            if not other_submissions or not submission_text:
                logger.warning("No text or other submissions for plagiarism check")
                return 0.0
            documents = [submission_text] + other_submissions
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).max() * 100

            submission_embedding = self.sentence_model.encode(submission_text, convert_to_tensor=True)
            other_embeddings = [self.sentence_model.encode(text, convert_to_tensor=True) for text in other_submissions]
            semantic_similarities = [util.cos_sim(submission_embedding, emb).item() * 100 for emb in other_embeddings]
            semantic_similarity = max(semantic_similarities) if semantic_similarities else 0.0

            score = min(100.0, 0.6 * tfidf_similarity + 0.4 * semantic_similarity)
            logger.info(f"Plagiarism score: {score:.2f}%")
            return score
        except Exception as e:
            logger.error(f"Error in plagiarism check: {str(e)}")
            return 0.0

    def grade_assignment(self, submission_text: str, assignment_keywords: str) -> tuple[float, float]:
        try:
            if not submission_text:
                logger.warning("No text provided for grading")
                return 0.0, 0.0

            tokens = nltk.word_tokenize(submission_text.lower())
            tokens = [t for t in tokens if t not in self.stop_words]
            
            # Length score (max 30 points)
            length_score = min(len(tokens) / 150, 1.0) * 30
            logger.info(f"Length score: {length_score:.2f} (tokens: {len(tokens)})")
            
            # FIX: Keyword score logic is corrected to prevent spamming and score inflation.
            # It now checks for the presence of unique keywords.
            keywords = [k.strip().lower() for k in assignment_keywords.split(',') if k.strip()] if assignment_keywords else []
            keyword_score = 0
            if keywords:
                found_keywords = set()
                for token in tokens:
                    for keyword in keywords:
                        if keyword not in found_keywords and (fuzz.ratio(token, keyword) > 80 or keyword in token):
                            found_keywords.add(keyword)
                            break # Move to the next token once a keyword match is found
                
                keyword_score = (len(found_keywords) / len(keywords)) * 30
                logger.info(f"Keyword score: {keyword_score:.2f} (matches: {len(found_keywords)}/{len(keywords)})")
            
            # Structure score (max 40 points)
            sentences = nltk.sent_tokenize(submission_text)
            # Avoid division by zero if there are no sentences
            avg_sentence_length = sum(len(nltk.word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0
            structure_score = min(len(sentences) / 4, 1.0) * 20
            structure_score += min(avg_sentence_length / 10, 1.0) * 20
            logger.info(f"Structure score: {structure_score:.2f} (sentences: {len(sentences)}, avg length: {avg_sentence_length:.2f})")
            
            # Content relevance score (max 10 points)
            academic_score = sum(1 for token in tokens if any(fuzz.ratio(token, term) > 80 for term in self.academic_terms)) / 10
            academic_score = min(academic_score, 1.0) * 10
            logger.info(f"Academic relevance score: {academic_score:.2f}")
            
            quality_score = length_score + keyword_score + structure_score + academic_score
            quality_score = min(quality_score, 100)
            
            # Relaxed penalty for diverse subjects
            if len(tokens) < 20 or len(sentences) < 1:
                quality_score *= 0.5
                logger.info("Applied penalty for very short or poorly structured content")
            
            grade = quality_score # Base grade is the quality score
            logger.info(f"Final quality score: {quality_score:.2f}")
            return quality_score, grade
        except Exception as e:
            logger.error(f"Error in grading: {str(e)}")
            return 0.0, 0.0

checker = AssignmentChecker()

# Authentication dependency
def get_current_user(username: str, role: str):
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row # Allows accessing columns by name
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND role = ?", (username, role))
        user = c.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return user

# API Endpoints
@app.post("/register")
async def register(user: UserCreate):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            # In a real app, hash the password. For a hackathon, this is okay.
            c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                      (user.username, user.password, user.role))
            conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/assignments")
async def create_assignment(assignment: AssignmentCreate, username: str = "professor1"):
    user = get_current_user(username, "professor")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO assignments (title, description, deadline, professor_id, keywords) VALUES (?, ?, ?, ?, ?)",
                      (assignment.title, assignment.description, assignment.deadline, user["id"], assignment.keywords))
            conn.commit()
        return {"message": "Assignment created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assignments")
async def get_assignments(username: str, role: str):
    user = get_current_user(username, role)
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            if role == "professor":
                c.execute("SELECT * FROM assignments WHERE professor_id = ?", (user["id"],))
            else:
                c.execute("SELECT * FROM assignments")
            assignments = c.fetchall()
        return [{"id": a["id"], "title": a["title"], "description": a["description"], "deadline": a["deadline"], "keywords": a["keywords"]} for a in assignments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submissions/{assignment_id}")
async def submit_assignment(assignment_id: int, file: UploadFile = File(...), username: str = "student1"):
    user = get_current_user(username, "student")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_content = await file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    await file.seek(0)
    
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM submissions WHERE assignment_id = ? AND student_id = ? AND file_hash = ?",
                  (assignment_id, user["id"], file_hash))
        if c.fetchone():
            raise HTTPException(status_code=400, detail="Duplicate submission detected")
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(UPLOAD_DIR, f"{assignment_id}_{user['id']}_{timestamp}_{file.filename}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        submission_text = checker.extract_text_from_pdf(file_path)
        
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT keywords, deadline FROM assignments WHERE id = ?", (assignment_id,))
            assignment_data = c.fetchone()
            if not assignment_data:
                raise HTTPException(status_code=404, detail="Assignment not found")
            keywords, deadline = assignment_data["keywords"], assignment_data["deadline"]
            
            c.execute("SELECT file_path FROM submissions WHERE assignment_id = ? AND student_id != ?",
                      (assignment_id, user["id"]))
            other_files = c.fetchall()
        
        other_texts = [checker.extract_text_from_pdf(f["file_path"]) for f in other_files]
        
        plagiarism_score = checker.check_plagiarism(submission_text, other_texts, file_path)
        quality_score, grade = checker.grade_assignment(submission_text, keywords)
        
        submission_time = datetime.now()
        deadline_time = datetime.fromisoformat(deadline)
        if submission_time > deadline_time:
            grade *= 0.8  # Apply 20% penalty for late submission
            logger.info(f"Applied 20% late submission penalty. New grade: {grade}")
        if plagiarism_score > 50: # Set a reasonable plagiarism threshold
            grade *= (1 - (plagiarism_score / 100)) # Penalize proportionally
            logger.info(f"Applied plagiarism penalty. New grade: {grade}")
        
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO submissions 
                        (assignment_id, student_id, file_path, file_hash, submission_time, grade, plagiarism_score, quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                      (assignment_id, user["id"], file_path, file_hash, submission_time.isoformat(), grade, plagiarism_score, quality_score))
            conn.commit()
            submission_id = c.lastrowid
        
        return SubmissionResponse(
            id=submission_id,
            assignment_id=assignment_id,
            student_id=user["id"],
            file_path=file_path,
            submission_time=submission_time.isoformat(),
            grade=grade,
            plagiarism_score=plagiarism_score,
            quality_score=quality_score
        )
    except Exception as e:
        logger.error(f"Error processing submission: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/submissions/{assignment_id}")
async def get_submissions(assignment_id: int, username: str, role: str):
    user = get_current_user(username, role)
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row # This is helpful for clarity
            c = conn.cursor()
            if role == "professor":
                c.execute("SELECT * FROM submissions WHERE assignment_id = ?", (assignment_id,))
            else:
                c.execute("SELECT * FROM submissions WHERE assignment_id = ? AND student_id = ?",
                          (assignment_id, user["id"]))
            submissions = c.fetchall()
        
        # CORRECTED: The indices are now correct, mapping the right DB column to the right Pydantic field.
        # Using sqlite3.Row (as dict keys) makes this even safer.
        return [SubmissionResponse(
            id=s["id"],
            assignment_id=s["assignment_id"],
            student_id=s["student_id"],
            file_path=s["file_path"],
            submission_time=s["submission_time"],
            grade=s["grade"],
            plagiarism_score=s["plagiarism_score"],
            quality_score=s["quality_score"]
        ) for s in submissions]
    except Exception as e:
        # Re-raise Pydantic errors properly for debugging
        if "validation error" in str(e):
            raise HTTPException(status_code=422, detail=str(e))
        logger.error(f"Error getting submissions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)