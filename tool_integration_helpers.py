"""
Tool Integration Helpers for MCP Research Bot
Provides intelligent tool flow suggestions and integration logic
"""

def get_tool_flow_suggestions(tools_used: list, response_content: str) -> dict:
    """
    Analyze tools used and response content to suggest optimal next tool flows
    """
    suggestions = {
        'immediate_next_steps': [],
        'exploration_paths': [],
        'deep_dive_options': []
    }
    
    # Tool flow logic based on what was just executed
    # Focused on metadata/abstract/summary-level exploration only
    tool_flows = {
        'search_papers': {
            'immediate': [
                'extract_info - Select a paper by ID for detailed information',
                'search_papers - Search related or narrower topics',
                'get_research_prompt - Get structured research guidance'
            ],
            'exploration': [
                'get_topic_papers - See if there are saved papers in this area',
                'get_available_folders - Explore what other topics are available'
            ],
            'deep_dive': [
                'extract_info - Get detailed info for specific papers',
                'search_papers - Find more papers on related topics',
                'get_topic_papers - Explore saved papers in depth'
            ]
        },
        'extract_info': {
            'immediate': [
                'search_papers - Find similar or citing papers',
                'get_topic_papers - Explore more papers in this topic area',
                'get_research_prompt - Get structured research guidance'
            ],
            'exploration': [
                'search_papers - Look for papers by the same authors',
                'get_available_folders - Explore other research areas'
            ],
            'deep_dive': [
                'extract_info - Get details for related papers',
                'search_papers - Find papers with similar abstracts',
                'get_topic_papers - Explore topic paper collections'
            ]
        },
        'get_topic_papers': {
            'immediate': [
                'extract_info - Select a paper by ID for detailed information',
                'search_papers - Find newer papers in this topic',
                'get_research_prompt - Get comprehensive research guidance'
            ],
            'exploration': [
                'get_available_folders - See what other topics are available',
                'search_papers - Explore related research areas'
            ],
            'deep_dive': [
                'extract_info - Get detailed info for topic papers',
                'search_papers - Find more papers in this research area',
                'get_research_prompt - Get topic-specific guidance'
            ]
        },
        'get_available_folders': {
            'immediate': [
                'get_topic_papers - Explore papers in interesting topics',
                'search_papers - Search for papers in promising areas',
                'get_research_prompt - Get research planning guidance'
            ],
            'exploration': [
                'get_topic_papers - Compare paper counts across topics',
                'search_papers - Explore interdisciplinary areas'
            ],
            'deep_dive': [
                'get_topic_papers - Explore topic paper collections',
                'search_papers - Find papers across multiple topics',
                'get_research_prompt - Get multi-topic research guidance'
            ]
        },
        'get_research_prompt': {
            'immediate': [
                'search_papers - Execute the research plan',
                'get_available_folders - See what topics are available',
                'get_topic_papers - Explore existing collections'
            ],
            'exploration': [
                'search_papers - Follow research guidance with searches',
                'extract_info - Get details on suggested papers'
            ],
            'deep_dive': [
                'search_papers - Execute detailed research steps',
                'get_topic_papers - Explore guided topic collections',
                'extract_info - Get detailed info on key papers'
            ]
        }
    }
    
    # Build suggestions based on tools used
    for tool in tools_used:
        if tool in tool_flows:
            suggestions['immediate_next_steps'].extend(tool_flows[tool]['immediate'])
            suggestions['exploration_paths'].extend(tool_flows[tool]['exploration'])
            suggestions['deep_dive_options'].extend(tool_flows[tool]['deep_dive'])
    
    # Remove duplicates while preserving order
    for key in suggestions:
        suggestions[key] = list(dict.fromkeys(suggestions[key]))
    
    return suggestions

def generate_tool_aware_fallback_questions(tools_used: list) -> list:
    """
    Generate fallback questions that are aware of which tools were just used
    Focused on metadata/abstract/summary-level exploration only
    """
    if not tools_used:
        return [
            "🔍 What research topic interests you?",
            "📁 What research areas are available?",
            "🗺️ How should I approach researching this?"
        ]
    
    fallback_map = {
        'search_papers': [
            "📋 Select a paper by ID for detailed information?",
            "🔍 Search for related research areas?",
            "🗺️ Get structured research guidance?"
        ],
        'extract_info': [
            "🔍 Find similar or related papers?",
            "📚 Explore more papers in this topic?",
            "📋 Get details for another paper?"
        ],
        'get_topic_papers': [
            "📋 Select a paper by ID for detailed information?",
            "🔍 Search for newer research?",
            "📚 Explore another topic area?"
        ],
        'get_available_folders': [
            "📚 Explore papers in specific topics?",
            "🔍 Search interesting research areas?",
            "🗺️ Get research planning guidance?"
        ],
        'get_research_prompt': [
            "🔍 Execute the research plan?",
            "📚 Explore existing paper collections?",
            "📁 See what topics are available?"
        ]
    }
    
    # Get fallback questions for the most recent tool used
    recent_tool = tools_used[-1] if tools_used else None
    if recent_tool in fallback_map:
        return fallback_map[recent_tool]
    
    # Default fallbacks
    return [
        "🔍 Search for more papers?",
        "📋 Get details about specific research?",
        "🗺️ Need research guidance?"
    ]

def predict_next_tool_from_question(question: str) -> str:
    """
    Predict which MCP tool a question is likely to trigger
    """
    question_lower = question.lower()
    
    # Tool prediction patterns
    tool_patterns = {
        'search_papers': [
            'search', 'find', 'look for', 'papers on', 'research on',
            'latest', 'recent', 'new papers', 'studies on'
        ],
        'extract_info': [
            'details', 'more about', 'specific paper', 'paper id',
            'arxiv', 'tell me about paper', 'information about'
        ],
        'get_topic_papers': [
            'saved', 'existing', 'collection', 'topic papers',
            'papers in', 'what papers do you have'
        ],
        'get_available_folders': [
            'available', 'topics', 'folders', 'what areas',
            'what subjects', 'research areas', 'categories'
        ],
        'get_research_prompt': [
            'guidance', 'how to research', 'research plan',
            'approach', 'methodology', 'how should i'
        ]
    }
    
    # Score each tool based on pattern matches
    tool_scores = {}
    for tool, patterns in tool_patterns.items():
        score = sum(1 for pattern in patterns if pattern in question_lower)
        if score > 0:
            tool_scores[tool] = score
    
    # Return the tool with highest score, or None if no clear match
    if tool_scores:
        return max(tool_scores.items(), key=lambda x: x[1])[0]
    
    return 'search_papers'  # Default assumption

def enhance_question_with_tool_context(question: str, predicted_tool: str) -> str:
    """
    Enhance a question with context that helps the LLM choose the right tool
    """
    tool_context_hints = {
        'search_papers': "Find research papers about: ",
        'extract_info': "Get detailed information about paper with ArXiv ID: ",
        'get_topic_papers': "Show me saved papers on topic: ",
        'get_available_folders': "What research topics are available? ",
        'get_research_prompt': "Give me research guidance for: "
    }
    
    # If the question doesn't already clearly indicate the tool, add context
    if predicted_tool in tool_context_hints:
        hint = tool_context_hints[predicted_tool]
        if not any(keyword in question.lower() for keyword in ['search', 'find', 'show', 'get', 'tell']):
            return f"{hint}{question}"
    
    return question
