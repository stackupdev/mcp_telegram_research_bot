"""
Simplified Tool Integration Helpers for MCP Research Bot
Essential functions for tool prediction and context enhancement
"""

def generate_tool_aware_fallback_questions(tools_used: list) -> list:
    """
    Generate fallback questions when LLM hint generation fails
    Focused on direct MCP tool usage
    """
    if not tools_used:
        return [
            "🔍 Search for papers on a specific topic",
            "📁 Browse available research topics",
            "🎯 Create a research plan for your topic"
        ]
    
    fallback_questions = []
    for tool in tools_used:
        if tool == 'search_papers':
            fallback_questions.extend([
                "🔍 Search for more papers on this topic",
                "📊 Compare these papers",
                "🎯 Get research guidance for this area"
            ])
        elif tool == 'extract_info':
            fallback_questions.extend([
                "📋 Get details for another paper",
                "🔍 Find similar papers",
                "📁 Browse papers in this topic"
            ])
        elif tool == 'get_topic_papers':
            fallback_questions.extend([
                "📊 Compare papers in this topic",
                "🔍 Search for newer papers",
                "📋 Get details for specific papers"
            ])
        elif tool == 'get_available_folders':
            fallback_questions.extend([
                "📁 Explore other research topics",
                "🔍 Search papers in new areas",
                "🎯 Create research plans for topics"
            ])
        elif tool == 'get_research_prompt':
            fallback_questions.extend([
                "🔍 Search papers using research guidance",
                "📁 Browse papers in planned areas",
                "📋 Get details for planned research"
            ])
    
    return fallback_questions[:4]

def predict_next_tool_from_question(question: str) -> str:
    """
    Predict which MCP tool a question is likely to trigger
    Simplified keyword matching for tool prediction
    """
    question_lower = question.lower().strip()
    
    # Simple keyword matching for tool prediction
    if any(keyword in question_lower for keyword in ['paper', 'detail', 'info', 'specific']):
        return 'extract_info'
    elif any(keyword in question_lower for keyword in ['search', 'find', 'look for', 'papers on']):
        return 'search_papers'
    elif any(keyword in question_lower for keyword in ['topic', 'papers in', 'collection', 'saved']):
        return 'get_topic_papers'
    elif any(keyword in question_lower for keyword in ['topics', 'folders', 'available', 'browse']):
        return 'get_available_folders'
    elif any(keyword in question_lower for keyword in ['plan', 'research plan', 'guidance', 'study']):
        return 'get_research_prompt'
    else:
        return 'search_papers'  # Default fallback

def enhance_question_with_tool_context(question: str, predicted_tool: str) -> str:
    """
    Enhance a question with context that helps the LLM choose the right tool
    Simplified context enhancement
    """
    tool_contexts = {
        'search_papers': "Search for academic papers by topic",
        'extract_info': "Get detailed paper information by ID",
        'get_topic_papers': "Browse papers in a specific topic",
        'get_available_folders': "List available research topics",
        'get_research_prompt': "Generate research guidance"
    }
    
    context = tool_contexts.get(predicted_tool, "Use MCP research tools")
    return f"{question} [{context}]"
