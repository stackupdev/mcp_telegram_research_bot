"""
Deep Research Logic Module

This module contains the core research functionality extracted from the MCP server
for direct integration into the Flask app and Telegram bot.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class ResearchProcessor:
    """Handles research data and process tracking."""
    
    def __init__(self):
        self.research_data = {
            "question": "",
            "elaboration": "",
            "subquestions": [],
            "search_results": {},
            "extracted_content": {},
            "final_report": "",
        }
        self.notes: List[str] = []

    def add_note(self, note: str):
        """Add a note to the research process."""
        self.notes.append(note)
        logger.debug(f"Note added: {note}")

    def update_research_data(self, key: str, value: Any):
        """Update a specific key in the research data dictionary."""
        self.research_data[key] = value
        self.add_note(f"Updated research data: {key}")

    def get_research_notes(self) -> str:
        """Return all research notes as a newline-separated string."""
        return "\n".join(self.notes)

    def get_research_data(self) -> Dict:
        """Return the current research data dictionary."""
        return self.research_data

    def clear_research(self):
        """Clear all research data and start fresh."""
        self.research_data = {
            "question": "",
            "elaboration": "",
            "subquestions": [],
            "search_results": {},
            "extracted_content": {},
            "final_report": "",
        }
        self.notes = []
        self.add_note("Research data cleared")


def generate_research_prompt(research_question: str) -> str:
    """
    Generate a structured research prompt for the given question.
    
    Args:
        research_question: The research question to investigate
        
    Returns:
        A formatted prompt string for conducting deep research
    """
    
    prompt_template = """
You are a professional researcher tasked with conducting thorough research on a topic and producing a structured, comprehensive report. Your goal is to provide a detailed analysis that addresses the research question systematically.

The research question is:

<research_question>
{research_question}
</research_question>

Follow these steps carefully:

1. <question_elaboration>
   Elaborate on the research question. Define key terms, clarify the scope, and identify the core issues that need to be addressed. Consider different angles and perspectives that are relevant to the question.
</question_elaboration>

2. <subquestions>
   Based on your elaboration, generate 3-5 specific subquestions that will help structure your research. Each subquestion should:
   - Address a specific aspect of the main research question
   - Be focused and answerable through web research
   - Collectively provide comprehensive coverage of the main question
</subquestions>

3. For each subquestion:
   a. <web_search_results>
      Search for relevant information using web search. For each subquestion, perform searches with carefully formulated queries.
      Extract meaningful content from the search results, focusing on:
      - Authoritative sources
      - Recent information when relevant
      - Diverse perspectives
      - Factual data and evidence
      
      Be sure to properly cite all sources and avoid extensive quotations. Limit quotes to less than 25 words each and use no more than one quote per source.
   </web_search_results>

   b. Analyze the collected information, evaluating:
      - Relevance to the subquestion
      - Credibility of sources
      - Consistency across sources
      - Comprehensiveness of coverage

4. Create a beautifully formatted research report as an artifact. Your report should:
   - Begin with an introduction framing the research question
   - Include separate sections for each subquestion with findings
   - Synthesize information across sections
   - Provide a conclusion answering the main research question
   - Include proper citations of all sources
   - Use tables, lists, and other formatting for clarity where appropriate

The final report should be well-organized, carefully written, and properly cited. It should present a balanced view of the topic, acknowledge limitations and areas of uncertainty, and make clear, evidence-based conclusions.

Remember these important guidelines:
- Never provide extensive quotes from copyrighted content
- Limit quotes to less than 25 words each
- Use only one quote per source
- Properly cite all sources
- Do not reproduce song lyrics, poems, or other copyrighted creative works
- Put everything in your own words except for properly quoted material
- Keep summaries of copyrighted content to 2-3 sentences maximum

Please begin your research process, documenting each step carefully.
"""
    
    return prompt_template.format(research_question=research_question)


def create_research_summary(research_question: str) -> str:
    """
    Create a simple research summary for Telegram display.
    
    Args:
        research_question: The research question
        
    Returns:
        A formatted summary for Telegram
    """
    return f"""🔍 **Deep Research Request**

**Question:** {research_question}

**Research Process:**
1. ✅ Question elaboration and scope definition
2. ✅ Generate focused subquestions  
3. ✅ Conduct web searches for each subquestion
4. ✅ Analyze and synthesize findings
5. ✅ Create comprehensive research report

**Instructions:**
This research prompt is designed for use with AI assistants that have web search capabilities. Copy the generated prompt and use it with Claude, ChatGPT, or similar AI tools to conduct thorough research on your topic.

**Expected Output:**
- Structured research report
- Proper source citations
- Analysis of multiple perspectives
- Clear conclusions and recommendations
"""


# Global research processor instance
research_processor = ResearchProcessor()
