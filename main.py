from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# 1. Import the NEW modern SDK
from google import genai

load_dotenv()

app = FastAPI(
    title="AI Career Analysis Agent",
    description="An AI-powered API that extracts structured career data from unstructured text."
)

# 2. Initialize the modern client
# It will automatically find the GEMINI_API_KEY in your .env file
client = genai.Client()

class DocumentRequest(BaseModel):
    resume_text: str

@app.post("/api/v1/analyze")
async def analyze_career_profile(req: DocumentRequest):
    prompt = f"""
    You are an expert AI Career Agent. Analyze the following unstructured resume text.
    Extract the key information and return it STRICTLY as a valid JSON object.
    Do not include markdown formatting or conversational text.
    
    The JSON must contain:
    - "primary_skills": A list of technical skills found.
    - "experience_level": A short string (e.g., "Junior", "Mid-Level").
    - "recommended_roles": A list of 3 specific job titles suited for this profile.

    Text to analyze:
    {req.resume_text}
    """
    
    try:
        # 3. Use the updated generation syntax
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return {"status": "success", "ai_analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))