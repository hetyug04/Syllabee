from dotenv import load_dotenv
load_dotenv() 

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional
import hashlib, os

from pydantic import BaseModel, Field, ValidationError, conlist
from langchain_google_genai import ChatGoogleGenerativeAI
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import re

import psycopg2
from psycopg2.extras import Json

from typing import Dict, Any

import numpy as np

from huggingface_hub import InferenceClient

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

hf_client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)


import logging

logger = logging.getLogger(__name__)

class GradeItem(BaseModel):
    title:  str
    weight: Optional[float] = Field(None, ge=0, le=100)
    due:    Optional[str]

ItemsList = Annotated[List[GradeItem], Field(min_items=1)]

class GradingCategory(BaseModel):
    items: ItemsList
    total_weight: Optional[float] = Field(None, ge=0, le=100)

class MetaData(BaseModel):
    school: str
    course: str
    course_code: Optional[str]
    professor: Optional[str]
    semester: Optional[str]
    delivery_method: Optional[str]
    meeting_info: Optional[Dict[str, Any]]
    credit_hours: Optional[int]
    contact_hours_per_week: Optional[int]
    outside_work_hours_per_week: Optional[float]
    office_hours: Optional[str]
    important_dates: List[Dict[str, Any]] = Field([], description="Dates of holidays, drop deadlines, etc.")
    prerequisites: List[str] = Field([], description="Course prerequisites.")
    learning_objectives: List[str] = Field([], description="Key learning goals.")
    policies: Dict[str, str] = Field({}, description="Named policy paragraphs (e.g., Late policy).")

    grading: Dict[str, GradingCategory] = Field(
        default_factory=dict,
        description="Dynamic map of *any* assessment category."
    )

    resources: Dict[str, Any] = Field({}, description="Textbooks, platforms, links, etc.")

class AgentState(TypedDict, total=False):
    user_id: str

    pdf_data: bytes
    ocr_text: str
    
    grading_policy: List[GradingCategory]
    important_dates: List[GradeItem]
    office_hours: Optional[str]
    professor: Optional[str]
    parsed_metadata: MetaData
    school: Optional[str]
    course_code: Optional[str]
    semester: Optional[str]
    delivery_method: Optional[str]
    meeting_info: Optional[dict]
    credit_hours: Optional[int]
    contact_hours_per_week: Optional[int]
    outside_work_hours_per_week: Optional[float]
    prerequisites: List[str]
    learning_objectives: List[str]
    ABET_relationship: Optional[str]
    required_texts: List[dict]
    platforms: List[str]
    academic_honesty_policy: Optional[str]
    academic_honesty_links: List[dict]
    generative_AI_policy: Optional[str]
    generative_AI_permissibility: Optional[str]
    generative_AI_acknowledgement_required: Optional[bool]
    generative_AI_consequences: Optional[str]
    stress_management_resources: List[dict]
    disability_services: Optional[dict]
    electronic_devices_policy: Optional[str]
    covid_policy: Optional[str]
    covid_policy_url: Optional[str]
    covid_policy_requirements: List[str]
    
    is_new: bool
    is_new_subscription: bool

STATE_CACHE: dict[str, dict] = {}
def ocr_node(state: AgentState) -> dict:
    logger.info("---OCR NODE---")
    from planner import ocr_from_bytes
    text = ocr_from_bytes(state["pdf_data"])
    logger.info("--- FULL OCR TEXT --")
    logger.info(text)
    return {"ocr_text": text}

def parse_syllabus(state: AgentState) -> dict:
    logger.info("---PARSE SYLLABUS NODE---")
    text = state["ocr_text"]
    uid = state["user_id"]

    key = f"{uid}:{hashlib.sha1(text.encode()).hexdigest()}"
    if key in STATE_CACHE:
        logger.info(f"---CACHE HIT for {key}---")
        cached_data = STATE_CACHE[key]
        return {k: v for k, v in cached_data.items() if k in AgentState.__annotations__}

    schema_str = json.dumps(MetaData.model_json_schema(), indent=2)

    messages = [
        SystemMessage(
            content=(
                "You are an expert syllabus parser and strict JSON generator. "
                "Your task is to extract all relevant information from the provided syllabus text "
                f"and format it precisely according to the following JSON schema:\n"
                f"{schema_str}\n\n"
                "Carefully read through the entire syllabus. "
                "A key piece of information to extract is the 'school' or 'university' name. "
                "Populate every field you can, prioritizing accuracy. "
                "If a specific field or array is not found in the syllabus, omit it or use a default empty list/null value as per the schema. "
                "For grading, find *all* assessment categories—homework, quizzes, midterms, finals, projects, labs, oral exams, etc.—and for each category produce a list of items with their titles and weights.  Do *not* assume fixed names: detect 'Final Exam', 'Mid-Term Exam', 'Quiz', etc., and include them in your JSON. "
                "For 'important_dates', include all relevant deadlines, exam dates, holidays, or specific weekly topics that imply a date. "
                "Ensure all date formats are consistent and recognizable (e.g., 'Month Dayth', 'YYYY-MM-DD'). "
                "If a percentage is found, convert it to a float (e.g., '20%' becomes 20.0). "
                "Extract all details for all fields, even if it requires combining information from different sections. "
                "If 'prerequisites' are 'N/A', represent it as ['N/A']."
            )
        ),
        HumanMessage(content=f"<doc>\n{text}\n</doc>")
    ]

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("PARSE_MODEL", "gemini-1.5-flash"),
        temperature=0,
        max_output_tokens=4096,
    )
    resp = llm.invoke(messages)
    raw = resp.content
    logger.info(f"THIS IS THE RAW RESPONSE FROM THE LLM:\n{raw}")

    clean = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE).strip()
    clean = re.sub(r"\n```$", "", clean).strip()

    try:
        metadata_obj = MetaData.model_validate_json(clean)
        parsed_data = metadata_obj.model_dump(exclude_unset=True)
    except ValidationError as e:
        logger.error(f"---LLM JSON VALIDATION ERROR---{e.json()}" )
        logger.error(f"---RAW LLM RESPONSE CAUSING ERROR---\n{raw}")
        raise RuntimeError(f"Bad LLM JSON: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"---LLM JSON DECODE ERROR---: {e}")
        logger.error(f"---RAW LLM RESPONSE CAUSING ERROR---\n{raw}")
        raise RuntimeError(f"LLM produced invalid JSON: {e}")
    except Exception as e:
        logger.error(f"---UNEXPECTED ERROR DURING PARSING---: {e}")
        logger.error(f"---RAW LLM RESPONSE CAUSING ERROR---\n{raw}")
        raise RuntimeError(f"An unexpected error occurred during parsing: {e}")

    state_update = {k: v for k, v in parsed_data.items() if k in AgentState.__annotations__}

    state_update = { k: v for k,v in parsed_data.items() if k in AgentState.__annotations__ }
    state_update.update({
      "user_id": state["user_id"],
      "pdf_data": state["pdf_data"],
      "ocr_text": state["ocr_text"],
      "parsed_metadata": metadata_obj,
    })


    STATE_CACHE[key] = state_update

    return state_update

def embed_with_hf(text: str) -> list[float]:
    output = hf_client.feature_extraction(text)
    return output.flatten().tolist()


def store_in_db(state: AgentState) -> dict:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur  = conn.cursor()

    
    md = state["parsed_metadata"]
    school = md.school
    course_code = md.course_code or md.course
    semester = md.semester
    professor = md.professor

    cur.execute("""
      SELECT id
        FROM syllabi
       WHERE school = %s
         AND course_code = %s
         AND semester    = %s
         AND professor   = %s
    """, (school, course_code, semester, professor))
    row = cur.fetchone()
    if row:
        syllabus_id = row[0]
    else:
        cur.execute("""
          INSERT INTO syllabi
            (school, course_code, semester, professor, metadata, pdf_hash)
          VALUES (%s, %s, %s, %s, %s, %s)
          RETURNING id
        """, (
          school,
          course_code,
          semester,
          professor,
          Json(md.model_dump()),
          hashlib.sha1(state["pdf_data"]).hexdigest()
        ))
        syllabus_id = cur.fetchone()[0]
        chunks = build_chunks(state["ocr_text"])
        for chunk in chunks:
            emb = embed_with_hf(chunk)
            cur.execute(
            """
            INSERT INTO course_chunks
                (syllabus_id, chunk_text, embedding, chunk_tsv)
            VALUES (%s, %s, %s, to_tsvector('english', %s))
            """,
            (syllabus_id, chunk, emb, chunk),
        )

    cur.execute("""
      INSERT INTO user_syllabi (user_id, syllabus_id)
      VALUES (%s, %s)
      ON CONFLICT (user_id, syllabus_id) DO NOTHING
    """, (state["user_id"], syllabus_id))
    is_new_subscription = (cur.rowcount == 1)

    conn.commit()
    cur.close()
    conn.close()

    return {"syllabus_id": syllabus_id,
            "is_new": not bool(row),
            "is_new_subscription": is_new_subscription,
            "school": school,
            "course_code": course_code,
            "professor": professor,
            "semester": semester}

def build_chunks(text: str, similarity_threshold: float = 0.95) -> List[str]:
    """
    Creates chunks based on semantic similarity between sentences.
    """
    logger.info("---Building chunks with semantic chunking...---")
    
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    if not sentences:
        return []

    sentence_embeddings = hf_client.feature_extraction(sentences)
    
    chunks = []
    current_chunk_sentences = [sentences[0]]
    
    for i in range(1, len(sentences)):
        prev_embedding = sentence_embeddings[i-1]
        current_embedding = sentence_embeddings[i]
        
        prev_embedding_norm = prev_embedding / np.linalg.norm(prev_embedding)
        current_embedding_norm = current_embedding / np.linalg.norm(current_embedding)
        
        similarity = np.dot(prev_embedding_norm, current_embedding_norm)
        
        if similarity >= similarity_threshold:
            current_chunk_sentences.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
            
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks

workflow = StateGraph(AgentState)
workflow.add_node("ocr", ocr_node)
workflow.add_node("parse_syllabus", parse_syllabus)
workflow.add_node("store_in_db", store_in_db)

workflow.set_entry_point("ocr")
workflow.add_edge("ocr", "parse_syllabus")
workflow.add_edge("parse_syllabus", "store_in_db")
workflow.add_edge("store_in_db", END)

app = workflow.compile()

class AskState(TypedDict, total=False):
    user_id: str
    query: str
    courses: List[Dict[str, Any]]
    course_code: Optional[str]
    syllabus_id: Optional[int]
    metadata: Optional[MetaData]
    chunks: Optional[List[str]]
    answer: Optional[str]
    clarification_needed: bool
    clarification_message: Optional[str]
    _next: Optional[str]

def router(state: AskState) -> dict:
    """
    Identifies the correct syllabus by leveraging an LLM for intelligent disambiguation.
    This method is simpler and more robust than complex keyword logic.
    """
    logger.info("---ROUTER---")
    user_id = state['user_id']
    query = state['query']

    if state.get("syllabus_id"):
        return {"syllabus_id": state["syllabus_id"], "_next": "load_context"}

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.course_code, s.metadata->>'course' as course_title, s.professor, s.semester
        FROM syllabi s
        JOIN user_syllabi us ON s.id = us.syllabus_id
        WHERE us.user_id = %s
    """, (user_id,))
    
    all_courses = [
        {"id": r[0], "course_code": r[1], "course_title": r[2] or "", "professor": r[3], "semester": r[4]}
        for r in cur.fetchall()
    ]
    cur.close()
    conn.close()

    if not all_courses:
        return {"clarification_needed": True, "clarification_message": "You don't have any syllabi uploaded.", "_next": "clarifier"}

    if len(all_courses) == 1:
        course = all_courses[0]
        logger.info(f"---Only one course available. Assuming question is for: {course['course_code']}---")
        return {"syllabus_id": course['id'], "_next": "load_context"}

    logger.info(f"---Multiple courses found. Using LLM to disambiguate for query: '{query}'---")
    
    course_list_for_prompt = "\n".join(
        [f"- Course ID: {c['id']}, Code: {c['course_code']}, Title: {c['course_title']}" for c in all_courses]
    )

    system_prompt = f"""
You are an intelligent routing agent. The user has asked a question and is subscribed to the following courses:
{course_list_for_prompt}

Analyze the user's question: "{query}"

Your task is to identify which course the user is asking about.
- If the question clearly refers to ONE of the courses, respond with *only* the integer Course ID for that course.
- If the question is ambiguous and could refer to more than one course (e.g., "my CS class"), respond with the exact word "ambiguous".
- If the question does not seem to relate to ANY of the listed courses, respond with the exact word "unrelated".
"""

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Based on my question, which course ID should I use? Please respond with only the ID or the required keyword.")
    ]
    response = llm.invoke(messages).content.strip() 

    logger.info(f"---LLM Router Response: '{response}'---")

    try:
        syllabus_id = int(response)
        if syllabus_id in [c['id'] for c in all_courses]:
            logger.info(f"---LLM routed to syllabus ID: {syllabus_id}---")
            return {"syllabus_id": syllabus_id, "_next": "load_context"}
        else:
            logger.info("---LLM returned an invalid ID. Defaulting to clarification.---")
            return {"courses": all_courses, "clarification_needed": True, "_next": "clarifier"}
    except (ValueError, TypeError):
        if response == "unrelated":
            return {
                "clarification_needed": True,
                "clarification_message": "I couldn't find a syllabus that relates to your question. Please try rephrasing.",
                "_next": "clarifier"
            }
        return {"courses": all_courses, "clarification_needed": True, "_next": "clarifier"}


def clarifier(state: AskState) -> dict:
    logger.info("---CLARIFIER---")
    courses = state.get("courses")
    
    if not courses:
        clarification_message = "You don't have any syllabi uploaded. Please upload a syllabus first."
    else:
        course_options = "\n".join([f"- {c['course_code']} ({c['semester']}) (id: {c['id']})" for c in courses])
        clarification_message = f"Which course are you asking about?\n{course_options}"
        
    return {
        "clarification_needed": True,
        "clarification_message": clarification_message,
        "courses": courses
    }

def load_context(state: AskState) -> dict:
    logger.info("---LOAD CONTEXT---")
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    
    cur.execute("SELECT metadata FROM syllabi WHERE id = %s", (state['syllabus_id'],))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No syllabus found for syllabus_id: {state['syllabus_id']}")
        
    metadata_json = row[0]
    metadata = MetaData.model_validate(metadata_json)
    
    cur.close()
    conn.close()
    
    return {"metadata": metadata}

def retriever(state: AskState) -> dict:
    logger.info("---RETRIEVER (Hybrid Search)---")
    
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    
    keyword_query = " | ".join(state['query'].split())
    
    cur.execute("""
        SELECT id, chunk_text, ts_rank(chunk_tsv, to_tsquery('english', %s)) as rank
        FROM course_chunks
        WHERE syllabus_id = %s AND chunk_tsv @@ to_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT 10;
    """, (keyword_query, state['syllabus_id'], keyword_query))
    keyword_results = cur.fetchall()

    query_embedding = embed_with_hf(state['query'])
    cur.execute("""
        SELECT id, chunk_text
        FROM course_chunks
        WHERE syllabus_id = %s
        ORDER BY embedding <-> %s::vector
        LIMIT 10;
    """, (state['syllabus_id'], repr(query_embedding)))
    vector_results = cur.fetchall()
    cur.close()
    conn.close()

    fused_scores = {}
    k = 60

    for i, row in enumerate(keyword_results):
        doc_id = row[0]
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {"score": 0, "text": row[1]}
        fused_scores[doc_id]["score"] += 1 / (k + i + 1)

    for i, row in enumerate(vector_results):
        doc_id = row[0]
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {"score": 0, "text": row[1]}
        fused_scores[doc_id]["score"] += 1 / (k + i + 1)
        
    sorted_fused_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    
    retrieved_chunks = [item["text"] for item in sorted_fused_results[:5]]
    
    return {"chunks": retrieved_chunks}

def generator(state: AskState) -> dict:
    logger.info("---GENERATOR---")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    context = (
        "Metadata:\n"
        + state['metadata'].model_dump_json(indent=2)
        + "\n\nRelevant Chunks:\n"
        + "\n".join(state['chunks'])
    )
    
    messages = [
        SystemMessage(content=f"You are a helpful course assistant. Answer the user's question based on the provided context. Provide a concise and well-formatted answer. Always use markdown to its fullest extent to best format responses in respect to the answers content. Use citations if necessary, like [Source: chunk text].\n\nContext:\n{context}"),
        HumanMessage(content=state['query'])
    ]
    
    response = llm.invoke(messages)
    return {"answer": response.content}

def log_qa_to_db(state: AskState) -> dict:
    logger.info("---LOGGER---")
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO qa_logs (user_id, course_id, query, answer)
        VALUES (%s, %s, %s, %s)
    """, (state['user_id'], state['syllabus_id'], state['query'], state['answer']))
    
    conn.commit()
    cur.close()
    conn.close()
    return {}

ask_graph_builder = StateGraph(AskState)

ask_graph_builder.add_node("router", router)
ask_graph_builder.add_node("clarifier", clarifier)
ask_graph_builder.add_node("load_context", load_context)
ask_graph_builder.add_node("retriever", retriever)
ask_graph_builder.add_node("generator", generator)
ask_graph_builder.add_node("logger", log_qa_to_db)

ask_graph_builder.set_entry_point("router")

ask_graph_builder.add_conditional_edges(
    "router",
    lambda s: s["_next"],
    {
        "clarifier": END,
        "load_context": "load_context"
    }
)

ask_graph_builder.add_edge("clarifier", END)

ask_graph_builder.add_edge("load_context", "retriever")
ask_graph_builder.add_edge("retriever", "generator")
ask_graph_builder.add_edge("generator", "logger")
ask_graph_builder.add_edge("logger", END)


ask_graph = ask_graph_builder.compile()

__all__ = ["app", "ask_graph"]
