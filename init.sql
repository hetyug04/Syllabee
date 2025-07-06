CREATE EXTENSION IF NOT EXISTS vector;

-- A central repository of all syllabi
CREATE TABLE syllabi (
  id          SERIAL PRIMARY KEY,
  school      TEXT    NOT NULL,
  course_code TEXT    NOT NULL,
  semester    TEXT    NOT NULL,
  professor   TEXT    NOT NULL,
  pdf_hash    TEXT    NOT NULL UNIQUE,
  metadata    JSONB   NOT NULL,
  created_at  TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE(school, course_code, semester, professor)
);

-- Link which users “have” which syllabus
CREATE TABLE user_syllabi (
  user_id    TEXT NOT NULL,
  syllabus_id INT NOT NULL REFERENCES syllabi(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, syllabus_id)
);

-- Store chunks keyed by syllabus, not by user
CREATE TABLE course_chunks (
  id          SERIAL PRIMARY KEY,
  syllabus_id INT NOT NULL REFERENCES syllabi(id) ON DELETE CASCADE,
  chunk_text  TEXT    NOT NULL,
  embedding   VECTOR(384) NOT NULL,
  chunk_tsv   TSVECTOR
);

-- Create a function to update the tsvector column
CREATE OR REPLACE FUNCTION update_chunk_tsv()
RETURNS TRIGGER AS $update_chunk_tsv$
BEGIN
  NEW.chunk_tsv := to_tsvector('english', NEW.chunk_text);
  RETURN NEW;
END;
$update_chunk_tsv$ LANGUAGE plpgsql;

-- Create a trigger to automatically update the tsvector column
CREATE TRIGGER tsvector_update
BEFORE INSERT OR UPDATE ON course_chunks
FOR EACH ROW EXECUTE FUNCTION update_chunk_tsv();

-- Log user questions and answers
CREATE TABLE qa_logs (
  id          SERIAL PRIMARY KEY,
  user_id     TEXT NOT NULL,
  course_id   INT NOT NULL REFERENCES syllabi(id) ON DELETE CASCADE,
  query       TEXT NOT NULL,
  answer      TEXT NOT NULL,
  timestamp   TIMESTAMP WITH TIME ZONE DEFAULT now()
);