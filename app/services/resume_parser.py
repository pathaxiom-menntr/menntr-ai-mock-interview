import PyPDF2
import docx
from typing import Optional

class ResumeParser:
    def parse(self, file_path: str) -> str:
        if file_path.endswith('.pdf'):
            return self._parse_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._parse_docx(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _parse_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def _parse_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

resume_parser_service = ResumeParser()
