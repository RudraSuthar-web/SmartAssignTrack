# SmartAssignTrack

SmartAssignTrack is a modern, automated web application designed to streamline the assignment management and grading process for educational environments. It provides distinct interfaces for professors and students, leveraging AI and machine learning to offer intelligent feedback on submissions. The system automatically assesses submission quality, checks for plagiarism, and calculates a final grade, providing instant results and reducing manual workload.

The application is built with a FastAPI backend for robust API services and a responsive vanilla JavaScript and Tailwind CSS frontend for a clean and intuitive user experience.

---

## Core Features

* **Role-Based Access**: Separate and secure views for Professors and Students.
* **Assignment Management**: Professors can create, view, and manage assignments, specifying details like title, description, deadline, and crucial keywords for grading.
* **PDF Submission**: Students can easily upload their assignments in the universally accepted PDF format.
* **Automated Grading & Quality Scoring**: Submissions are automatically graded based on a comprehensive quality score that evaluates:
    * **Content-Length & Structure**: Analyzes the number of tokens and sentences.
    * **Keyword Relevance**: Matches the submission's content against professor-defined keywords using fuzzy matching.
    * **Academic Language**: Identifies the use of academic and technical terminology.
* **Advanced Plagiarism Detection**:
    * Utilizes a hybrid approach combining **TF-IDF** for lexical similarity and **Sentence-Transformers** for semantic meaning.
    * Compares each new submission against all previous submissions for the same assignment to ensure academic integrity.
* **Dynamic Grade Penalties**: The final grade is intelligently adjusted based on:
    * **Late Submissions**: A penalty is applied if the submission is past the deadline.
    * **Plagiarism Score**: The grade is proportionally reduced if the plagiarism score exceeds a set threshold.
* **Real-time Dashboards & Feedback**:
    * **For Students**: An updated list of their submissions with detailed scores (Quality, Plagiarism, Final Grade) is available immediately after submission.
    * **For Professors**: A centralized modal to view all submissions for a specific assignment, allowing for quick assessment and overview.
* **Responsive & Professional UI**:
    * Built with Tailwind CSS for a clean, modern, and responsive design that works on any device.
    * Features a professional black-and-white theme with intuitive loading indicators and notifications.

---

## Tech Stack

| Category           | Technology / Library                                       |
| ------------------ | ---------------------------------------------------------- |
| **Backend** | Python, FastAPI, Uvicorn                                  |
| **Database** | SQLite (for simplicity and portability)                    |
| **Frontend** | HTML, Vanilla JavaScript, Tailwind CSS                     |
| **Text Extraction**| PyTesseract (OCR), PyPDF2, pdf2image                      |
| **ML & NLP** | Scikit-learn, Sentence-Transformers, NLTK, FuzzyWuzzy    |
| **Dependencies** | Pydantic, python-multipart, python-Levenshtein           |

---

## Setup and Installation

Follow these steps to get the application running locally on your machine.

### Prerequisites

1.  **Python 3.7+**: Ensure you have a modern version of Python installed.
2.  **Pip**: Python's package installer.
3.  **Tesseract OCR Engine**: This is **crucial** for extracting text from PDF files.
    * **Windows**: Download and run the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page. **Important**: Make sure to add the Tesseract installation directory to your system's `PATH` environment variable.
    * **macOS**: `brew install tesseract`
    * **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd SmartAssignTrack