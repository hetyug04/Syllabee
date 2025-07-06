from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pytesseract
from pdf2image import convert_from_bytes
import pdfplumber
from io import BytesIO
import numpy as np
import os
from typing import List, Optional
from langgraph_flow import app as langgraph_app, ask_graph
import psycopg2
import re

import logging

logger = logging.getLogger(__name__)

app = FastAPI()

class AskPayload(BaseModel):
    user_id: str
    query: str
    syllabus_id: Optional[int] = None

class AskResponse(BaseModel):
    answer: Optional[str] = None
    clarification_needed: bool = False
    clarification_message: Optional[str] = None
    courses: Optional[List[dict]] = None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(
            status_code=400,
            content={"message": "Invalid file type. Only PDFs are allowed."},
        )

    try:
        pdf_data = await file.read()

        MAX_FILE_SIZE = 20 * 1024 * 1024
        if len(pdf_data) > MAX_FILE_SIZE:
            logger.warning(f"File size exceeds limit: {file.filename}")
            return JSONResponse(
                status_code=413,
                content={"message": "File size exceeds the 20MB limit."},
            )

        initial_state = {
            "user_id": user_id,
            "pdf_data": pdf_data,
        }
        final_state = langgraph_app.invoke(initial_state)
        details = final_state

        if 'pdf_data' in final_state:
            del final_state['pdf_data']

        logger.info(f"Syllabus processed successfully for user {user_id}: {details.get('course_code')}")
        return {
            "message": "Syllabus processed successfully.",
            "details": {
                "is_new":      details.get("is_new"),
                "is_new_subscription": details.get("is_new_subscription"),
                "school":      details.get("school"),
                "course_code": details.get("course_code"),
                "professor":   details.get("professor"),
                "semester":    details.get("semester"),
            }
        }
    except Exception as e:
        logger.error(f"An error occurred during file upload and processing: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "An unexpected error occurred. Please try again later."},
        )

@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskPayload):
    initial_state = {
        "user_id": payload.user_id,
        "query": payload.query,
        "syllabus_id": payload.syllabus_id,
    }
    
    final_state = ask_graph.invoke(initial_state)
    
    if final_state.get("clarification_needed"):
        return AskResponse(
            clarification_needed=True,
            clarification_message=final_state.get("clarification_message"),
            courses=final_state.get("courses")
        )
    
    return AskResponse(answer=final_state.get("answer"))


@app.get("/list")
async def list_courses(user_id: str):
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.school, s.course_code, s.professor, s.semester
        FROM syllabi s
        JOIN user_syllabi us ON s.id = us.syllabus_id
        WHERE us.user_id = %s
        ORDER BY s.school, s.semester
    """, (user_id,))
    
    courses_by_school = {}
    for row in cur.fetchall():
        school = row[1]
        semester = row[4]
        course_info = f"{row[2]} - Prof. {row[3]} (ID: {row[0]})"
        if school not in courses_by_school:
            courses_by_school[school] = {}
        if semester not in courses_by_school[school]:
            courses_by_school[school][semester] = []
        courses_by_school[school][semester].append(course_info)

    cur.close()
    conn.close()
    return {"courses_by_school": courses_by_school}


@app.get("/list_all")
async def list_all_courses():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("""
        SELECT id, school, course_code, professor, semester
        FROM syllabi
        ORDER BY school, semester, course_code
    """)
    
    courses_by_school_semester_subject = {}
    for row in cur.fetchall():
        school = row[1]
        semester = row[4]
        course_code = row[2]
        match = re.match(r"([A-Z]+)", course_code)
        subject = match.group(1) if match else "Unknown"
        course_info = f"{course_code} - Prof. {row[3]} (ID: {row[0]})"

        if school not in courses_by_school_semester_subject:
            courses_by_school_semester_subject[school] = {}
        if semester not in courses_by_school_semester_subject[school]:
            courses_by_school_semester_subject[school][semester] = {}
        if subject not in courses_by_school_semester_subject[school][semester]:
            courses_by_school_semester_subject[school][semester][subject] = []
            
        courses_by_school_semester_subject[school][semester][subject].append(course_info)

    cur.close()
    conn.close()
    return {"courses_by_school_semester_subject": courses_by_school_semester_subject}



@app.post("/subscribe")
async def subscribe(user_id: str, syllabus_id: int):
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO user_syllabi (user_id, syllabus_id)
            VALUES (%s, %s)
            ON CONFLICT (user_id, syllabus_id) DO NOTHING
        """, (user_id, syllabus_id))
        conn.commit()
        if cur.rowcount > 0:
            return {"message": "Subscribed successfully."}
        else:
            return {"message": "You are already subscribed to this syllabus."}
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(status_code=404, detail="Syllabus not found.") from e
    finally:
        cur.close()
        conn.close()

@app.post("/unsubscribe")
async def unsubscribe(user_id: str, syllabus_id: int):
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM user_syllabi
        WHERE user_id = %s AND syllabus_id = %s
    """, (user_id, syllabus_id))
    conn.commit()
    if cur.rowcount > 0:
        message = "Unsubscribed successfully."
    else:
        message = "You were not subscribed to this syllabus."
    cur.close()
    conn.close()
    return {"message": message} 


@app.post("/plan")
async def plan_endpoint(file: UploadFile = File(...)):
    bytes = await file.read()
    text = ocr_from_bytes(bytes)
    return text

def ocr_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts text from a PDF's bytes using a robust hybrid approach.
    
    1. It first tries fast, direct text extraction with pdfplumber.
    2. If that yields no text, it falls back to converting the PDF to 
       images and using Tesseract OCR, which handles scanned documents.
    """
    logger.info("---Attempting direct text extraction with pdfplumber...---")
    text = ""
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text(x_tolerance=1)
                if page_text:
                    text += page_text + f"\n\n--- PAGE {i} ---\n\n"
        
        if len(text.strip()) > 100:
            logger.info("---Success: Extracted text directly.---")
            return text
    except Exception as e:
        logger.warning(f"---Direct extraction failed ({e}). Proceeding to OCR fallback.---")
        pass

    logger.info("---Executing OCR fallback with Tesseract.---")
    text = ""
    try:
        images = convert_from_bytes(pdf_bytes)
        
        for i, image in enumerate(images, 1):
            page_text = pytesseract.image_to_string(image)
            if page_text:
                text += page_text + f"\n\n--- PAGE {i} ---\n\n"
        
        logger.info("---Success: Extracted text with OCR.---")
        return text

    except Exception as e:
        logger.error(f"---FATAL: OCR fallback failed: {e}---")
        raise RuntimeError(
            "The PDF could not be read. This may be a scanned document, "
            "and the OCR engine (Tesseract/Poppler) may not be installed correctly. "
            f"Error: {e}"
        )
