import os
import json
import asyncio
import re
from typing import List
from flask import Flask, render_template, request, jsonify
from groq import Groq
from telegram import Update, Bot, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, CallbackQueryHandler, Filters
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession



# Flask app initialization
app = Flask(__name__)

# Telegram bot configuration
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_BOT = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
telegram_dispatcher = Dispatcher(TELEGRAM_BOT, None, workers=0, use_context=True) if TELEGRAM_BOT else None

# In-memory user data for session context per user
user_data = {}

# Research papers directory
PAPER_DIR = "papers"

# Maximum tokens to allow in conversation history before truncating
MAX_TOKENS = 4000

RESEARCH_SERVER_URL = "https://mcp-arxiv-server.onrender.com/sse"

# MCP client will be created as needed
mcp_client_factory = None

def init_mcp_client():
    """Initialize the MCP client factory"""
    global mcp_client_factory
    try:
        mcp_client_factory = lambda: sse_client(RESEARCH_SERVER_URL)
        print(f"MCP client factory initialized successfully for {RESEARCH_SERVER_URL}")
    except Exception as e:
        print(f"Failed to initialize MCP client factory: {e}")
        mcp_client_factory = None

# Initialize MCP client factory at startup
init_mcp_client()

async def call_mcp_tool(tool_name: str, **kwargs):
    """Call an MCP tool with proper async handling"""
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return None
    
    try:
        async with mcp_client_factory() as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=kwargs)
                return result
    except Exception as e:
        print(f"Error calling MCP tool {tool_name}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

async def read_mcp_resource(uri: str):
    """Read an MCP resource with proper async handling"""
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return None
    
    try:
        async with mcp_client_factory() as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.read_resource(uri)
                return result
    except Exception as e:
        print(f"Error reading MCP resource {uri}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

# ============================================================================
# RESEARCH FUNCTIONALITY (via remote MCP server with SSE)
# ============================================================================

def search_papers(topic: str, max_results: int = 10) -> List[str]:
    """
    Call the remote MCP server to search for papers on a topic.
    Returns: List of paper IDs
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return []
    
    try:
        # Call the search_papers tool via MCP using async helper
        print(f"Calling search_papers tool with topic='{topic}', max_results={max_results}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_mcp_tool("search_papers", topic=topic, max_results=max_results))
        loop.close()
        
        print(f"Search result: {result}")
        print(f"Search result type: {type(result)}")
        
        # Handle MCP tool call result
        if hasattr(result, 'content') and result.content:
            # Extract content from MCP tool result
            content = result.content
            print(f"Extracted content: {content}")
            if isinstance(content, list):
                print(f"Found {len(content)} papers")
                # Extract text from TextContent objects
                paper_ids = []
                for item in content:
                    if hasattr(item, 'text'):
                        paper_ids.append(item.text)
                    elif isinstance(item, str):
                        paper_ids.append(item)
                    else:
                        paper_ids.append(str(item))
                return paper_ids
            elif isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
                return [content] if content else []
        elif isinstance(result, list):
            print(f"Found {len(result)} papers")
            return result
        elif isinstance(result, str):
            print(f"Got string result: {result}")
            # Try to parse if it's a JSON string
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            return [result] if result else []
        elif result is None:
            print("Got None result from MCP tool")
            return []
        else:
            print(f"Unexpected result type: {type(result)}")
            return []
    except Exception as e:
        print(f"Error calling search_papers via MCP: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

def extract_info(paper_id: str):
    """
    Call the remote MCP server to get info about a specific paper.
    Returns: Paper info as dict or error message
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return {"error": "MCP client not available"}
    
    try:
        # Call the extract_info tool via MCP using async helper
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_mcp_tool("extract_info", paper_id=paper_id))
        loop.close()
        
        # Handle CallToolResult objects
        if hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                # Extract text from first TextContent object
                first_item = content[0]
                if hasattr(first_item, 'text'):
                    text_content = first_item.text
                else:
                    text_content = str(first_item)
            else:
                text_content = str(content)
        elif isinstance(result, str):
            text_content = result
        elif result is None:
            return {"error": "No result from MCP tool"}
        else:
            text_content = str(result)
        
        # Check for error messages
        if isinstance(text_content, str) and text_content.startswith("There's no saved information"):
            return {"error": text_content}
        
        # Try to parse as JSON
        if isinstance(text_content, str):
            try:
                return json.loads(text_content)
            except json.JSONDecodeError:
                return {"error": text_content}
        
        return {"error": "Unexpected result format"}
    except Exception as e:
        print(f"Error calling extract_info via MCP: {e}")
        return {"error": str(e)}

def get_available_folders():
    """
    Call the remote MCP server to list all available topic folders.
    Returns: List of topic folder names
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return []
    
    try:
        # Get the folders resource via MCP using async helper
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(read_mcp_resource("papers://folders"))
        loop.close()
        
        # Handle MCP ReadResourceResult object
        text_content = None
        if hasattr(result, 'contents') and result.contents:
            # Extract text from the first content item
            first_content = result.contents[0]
            if hasattr(first_content, 'text'):
                text_content = first_content.text
        elif isinstance(result, str):
            text_content = result
        
        if text_content:
            lines = text_content.split('\n')
            folders = []
            for line in lines:
                if line.strip().startswith('- '):
                    folder_name = line.strip()[2:]  # Remove '- ' prefix
                    folders.append(folder_name)
            return folders
        elif isinstance(result, dict):
            return list(result.keys()) if result else []
        elif isinstance(result, list):
            return result
        elif result is None:
            return []
        else:
            return []
    except Exception as e:
        print(f"Error calling get_available_folders via MCP: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

def get_topic_papers(topic: str):
    """
    Call the remote MCP server to get all papers for a topic.
    Returns: List of paper dictionaries
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return []
    
    try:
        # Get the topic papers resource via MCP using async helper
        topic_formatted = topic.lower().replace(" ", "_")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(read_mcp_resource(f"papers://{topic_formatted}"))
        loop.close()
        
        # Handle MCP ReadResourceResult object
        text_content = None
        if hasattr(result, 'contents') and result.contents:
            # Extract text from the first content item
            first_content = result.contents[0]
            if hasattr(first_content, 'text'):
                text_content = first_content.text
        elif isinstance(result, str):
            text_content = result
        
        if not text_content:
            return []
            
        # Check for no papers message
        if "No papers found" in text_content or "No topics found" in text_content:
            return []
        
        # Parse the markdown content to extract paper information
        papers = []
        lines = text_content.split('\n')
        current_paper = {}
        in_summary = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('## ') and not line.startswith('## Papers on'):
                # New paper title
                if current_paper and 'title' in current_paper:
                    papers.append(current_paper)
                current_paper = {'title': line[3:]}
                in_summary = False
                
            elif line.startswith('- **Paper ID**:'):
                paper_id = line.split(':', 1)[1].strip()
                current_paper['id'] = paper_id
                current_paper['entry_id'] = f"https://arxiv.org/abs/{paper_id}"
                in_summary = False
                
            elif line.startswith('- **Authors**:'):
                authors_str = line.split(':', 1)[1].strip()
                current_paper['authors'] = [author.strip() for author in authors_str.split(',')]
                in_summary = False
                
            elif line.startswith('- **Published**:'):
                current_paper['published'] = line.split(':', 1)[1].strip()
                in_summary = False
                
            elif line.startswith('- **PDF URL**:'):
                # Extract URL from markdown link format [url](url)
                url_part = line.split(':', 1)[1].strip()
                if '[' in url_part and '](' in url_part:
                    current_paper['pdf_url'] = url_part.split('](')[1].rstrip(')')
                else:
                    current_paper['pdf_url'] = url_part
                in_summary = False
                    
            elif line.startswith('### Summary'):
                # Start collecting summary lines
                in_summary = True
                current_paper['summary'] = ''
                continue
                
            elif in_summary and line:
                # Collect all summary lines - be more permissive about what we collect
                # Stop only on clear section boundaries
                if line.startswith('## ') and not line.startswith('## Papers on'):
                    # This is a new paper, stop collecting summary
                    in_summary = False
                    # Process this line as a new paper title
                    if current_paper and 'title' in current_paper:
                        papers.append(current_paper)
                    current_paper = {'title': line[3:]}
                elif line.startswith('---') or line.startswith('Total papers:'):
                    # End of section
                    in_summary = False
                else:
                    # Continue collecting summary text
                    if 'summary' not in current_paper:
                        current_paper['summary'] = ''
                    if current_paper['summary']:
                        current_paper['summary'] += ' '
                    current_paper['summary'] += line.replace('...', '').strip()
                
            elif line.startswith('## ') and not line.startswith('## Papers on'):
                # New paper title (handle case where we're not in summary mode)
                if not in_summary:
                    if current_paper and 'title' in current_paper:
                        papers.append(current_paper)
                    current_paper = {'title': line[3:]}
                in_summary = False
                
            elif line.startswith('---') or line.startswith('Total papers:'):
                # End of current paper or section
                in_summary = False
        
        # Don't forget the last paper
        if current_paper and 'title' in current_paper:
            papers.append(current_paper)
        
        return papers
    except Exception as e:
        print(f"Error calling get_topic_papers via MCP: {e}")
        return []

def get_research_prompt(topic: str, num_papers: int = 10) -> str:
    """
    Get a structured research prompt from the MCP server for enhanced AI research.
    This uses the MCP prompt primitive to generate comprehensive research instructions.
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return f"Search for {num_papers} academic papers about '{topic}' and provide a comprehensive analysis."
    
    try:
        # Access the generate_search_prompt via MCP
        print(f"Getting research prompt for topic: {topic}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_mcp_tool("generate_search_prompt", topic=topic, num_papers=num_papers))
        loop.close()
        
        print(f"Prompt result: {result}")
        
        # Extract prompt content
        if hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if hasattr(first_item, 'text'):
                    return first_item.text
                else:
                    return str(first_item)
            else:
                return str(content)
        elif isinstance(result, str):
            return result
        else:
            # Fallback to basic prompt
            return f"Search for {num_papers} academic papers about '{topic}' and provide a comprehensive analysis."
            
    except Exception as e:
        print(f"Error getting research prompt via MCP: {e}")
        # Fallback to basic prompt
        return f"Search for {num_papers} academic papers about '{topic}' and provide a comprehensive analysis."

# ============================================================================
# FUNCTION CALLING INFRASTRUCTURE
# ============================================================================

# Define function schemas for MCP tools
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for academic papers on ArXiv by topic. Use this when the user asks about research papers, recent studies, or wants to find papers on a specific topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research topic to search for (e.g., 'quantum computing', 'machine learning', 'neural networks')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to return (default: 10, max: 10)",
                        "default": 10
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": "Get detailed information about a specific paper using its ArXiv ID. Use this when the user wants more details about a specific paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The ArXiv paper ID (e.g., '2301.12345')"
                    }
                },
                "required": ["paper_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic_papers",
            "description": "Get all papers that have been previously saved for a specific research topic. Use this to retrieve papers from a known research area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic name to get papers for (use exact topic names from get_available_folders)"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_folders",
            "description": "List all available research topic folders that contain saved papers. Use this to see what research topics are available.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_research_prompt",
            "description": "Generate a comprehensive research prompt for in-depth academic analysis. Use this when you want to provide structured, detailed research guidance for a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research topic to generate a comprehensive prompt for"
                    },
                    "num_papers": {
                        "type": "integer",
                        "description": "Number of papers to include in the research analysis (default: 10)",
                        "default": 10
                    }
                },
                "required": ["topic"]
            }
        }
    }
]

def execute_function_call(function_name: str, arguments: dict):
    """
    Execute a function call by routing to the appropriate MCP tool.
    Returns the result of the function call with robust error handling.
    """
    try:
        print(f"Executing function call: {function_name} with arguments: {arguments}")
        
        # Validate inputs
        if not function_name:
            return {
                "success": False,
                "function": "unknown",
                "error": "Function name is empty"
            }
            
        if not isinstance(arguments, dict):
            arguments = {}
        
        if function_name == "search_papers":
            topic = arguments.get("topic")
            if not topic or not isinstance(topic, str):
                return {
                    "success": False,
                    "function": function_name,
                    "error": "Topic parameter is required and must be a string"
                }
            
            max_results = arguments.get("max_results", 10)
            try:
                max_results = int(max_results)
                max_results = max(1, min(max_results, 10))  # Clamp between 1-10
            except (ValueError, TypeError):
                max_results = 10
            
            result = search_papers(topic, max_results)
            
            # Validate result
            if result is None:
                result = []
            elif not isinstance(result, list):
                result = [str(result)] if result else []
            
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Found {len(result)} papers on '{topic}'"
            }
            
        elif function_name == "extract_info":
            paper_id = arguments.get("paper_id")
            if not paper_id or not isinstance(paper_id, str):
                return {
                    "success": False,
                    "function": function_name,
                    "error": "Paper ID parameter is required and must be a string"
                }
            
            result = extract_info(paper_id)
            if not result or (isinstance(result, dict) and "error" in result):
                error_msg = result.get("error", "Failed to retrieve paper information") if isinstance(result, dict) else "Failed to retrieve paper information"
                return {
                    "success": False,
                    "function": function_name,
                    "error": error_msg
                }
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Retrieved details for paper {paper_id}"
            }
            
        elif function_name == "get_topic_papers":
            topic = arguments.get("topic")
            if not topic or not isinstance(topic, str):
                return {
                    "success": False,
                    "function": function_name,
                    "error": "Topic parameter is required and must be a string"
                }
            
            result = get_topic_papers(topic)
            if result is None:
                result = []
            elif not isinstance(result, list):
                result = []
            
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Found {len(result)} saved papers for topic '{topic}'"
            }
            
        elif function_name == "get_available_folders":
            result = get_available_folders()
            if result is None:
                result = []
            elif not isinstance(result, list):
                result = []
            
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Found {len(result)} available research topics"
            }
            
        elif function_name == "get_research_prompt":
            topic = arguments.get("topic")
            if not topic or not isinstance(topic, str):
                return {
                    "success": False,
                    "function": function_name,
                    "error": "Topic parameter is required and must be a string"
                }
            
            num_papers = arguments.get("num_papers", 10)
            if not isinstance(num_papers, int) or num_papers < 1:
                num_papers = 10
            
            result = get_research_prompt(topic, num_papers)
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Generated comprehensive research prompt for '{topic}' with {num_papers} papers"
            }
            
        else:
            return {
                "success": False,
                "function": function_name,
                "error": f"Unknown function: {function_name}"
            }
            
    except Exception as e:
        print(f"Error executing function {function_name}: {e}")
        return {
            "success": False,
            "function": function_name,
            "error": str(e)
        }

# ============================================================================
# LLM FUNCTIONALITY (Groq API) - Enhanced with Function Calling
# ============================================================================

def get_llama_reply(messages: list, enable_tools: bool = True, update=None) -> str:
    """
    Enhanced LLama reply function with function calling support.
    Animation is triggered when tools are actually called.
    """
    try:
        client = Groq()
        
        # Prepare the API call parameters
        api_params = {
            "model": "llama-3.1-8b-instant",
            "messages": messages
        }
        
        # Add tools if enabled
        if enable_tools:
            api_params["tools"] = MCP_TOOLS
            api_params["tool_choice"] = "auto"
        
        completion = client.chat.completions.create(**api_params)
        message = completion.choices[0].message
        
        # Check if the model wants to call functions
        if message.tool_calls:
            total_tools = len(message.tool_calls)
            print(f"LLama wants to call {total_tools} function(s)")
            
            # Add the assistant's message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in message.tool_calls]
            })
            
            # Execute each tool call with progressive feedback
            success_count = 0
            for i, tool_call in enumerate(message.tool_calls, 1):
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                # Send starting status
                if update:
                    details = f"({i}/{total_tools})"
                    if arguments.get('topic'):
                        details += f" for '{arguments['topic']}'"
                    elif arguments.get('paper_id'):
                        details += f" for paper {arguments['paper_id']}"
                    send_progressive_research_update(update, function_name, 'starting', details)
                
                # Execute the function
                function_result = execute_function_call(function_name, arguments)
                
                # Send completion status
                if update:
                    if function_result.get('success'):
                        success_count += 1
                        result_details = function_result.get('summary', 'completed successfully')
                        send_progressive_research_update(update, function_name, 'completed', result_details)
                    else:
                        error_msg = function_result.get('error', 'unknown error')
                        send_progressive_research_update(update, function_name, 'completed', f"Error: {error_msg}")
                
                # Add the function result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_result)
                })
            
            # Get the final response from the model with function results
            final_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages
            )
            
            # Send final completion message
            if update:
                finalize_research_feedback(update, total_tools, success_count)
            
            final_response = final_completion.choices[0].message.content
            
            # Validate final response
            if not final_response or not final_response.strip():
                return "I found some research results but had trouble formatting the response. Please try asking your question again."
            
            # Format thinking tags nicely
            return format_deepseek_thinking(final_response)
        
        # No function calls, return regular response
        response = message.content
        
        # Validate regular response
        if not response or not response.strip():
            return "I'm having trouble generating a response right now. Please try asking your question again."
            
        # Format thinking tags nicely
        return format_deepseek_thinking(response)
        
    except Exception as e:
        error_str = str(e)
        print(f"Error in get_llama_reply: {error_str}")
        
        # Handle function calling errors
        if "tool_use_failed" in error_str or "Failed to call a function" in error_str:
            return "I tried to search for research information but encountered a technical issue with the function calling system. Please try rephrasing your question or ask me something else."
        
        # Handle token limit errors
        if "413" in error_str and "Request too large" in error_str:
            return "⚠️ Your conversation history is too long for the model's token limit. Please use /reset to start a new conversation, or ask a shorter question."
        
        return f"⚠️ Error from Groq API: {error_str}"

# Global dictionary to track progressive research feedback states
animation_states = {}

def send_progressive_research_update(update, tool_name, status, details=None):
    """
    Send progressive research feedback with specific tool status updates
    """
    if update is None:
        return
    
    # Handle different input types
    if hasattr(update, 'effective_chat'):
        chat_id = update.effective_chat.id
        bot = update.effective_chat.bot
    elif isinstance(update, int):
        chat_id = update
        bot = TELEGRAM_BOT
        if not bot:
            return
    else:
        return
    
    # Tool-specific status messages
    tool_messages = {
        'search_papers': {
            'starting': '🔍 Searching ArXiv papers',
            'processing': '📄 Analyzing search results',
            'completed': '✅ Found papers'
        },
        'extract_info': {
            'starting': '📋 Extracting paper details',
            'processing': '🔬 Analyzing paper content',
            'completed': '✅ Paper information extracted'
        },
        'get_topic_papers': {
            'starting': '📚 Loading saved papers',
            'processing': '📖 Organizing paper collection',
            'completed': '✅ Papers loaded'
        },
        'get_available_folders': {
            'starting': '📁 Scanning research topics',
            'processing': '🗂️ Organizing topic list',
            'completed': '✅ Topics loaded'
        },
        'get_research_prompt': {
            'starting': '💡 Generating research prompt',
            'processing': '✍️ Structuring research guidance',
            'completed': '✅ Research prompt ready'
        }
    }
    
    # Get appropriate message
    tool_msgs = tool_messages.get(tool_name, {
        'starting': f'🔧 Executing {tool_name}',
        'processing': f'⚙️ Processing {tool_name}',
        'completed': f'✅ {tool_name} completed'
    })
    
    base_message = tool_msgs.get(status, f'🔧 {tool_name}: {status}')
    
    # Add details if provided
    if details:
        message = f"{base_message} - {details}"
    else:
        message = base_message
    
    # Update existing animation message if active
    if chat_id in animation_states and animation_states[chat_id].get('active'):
        try:
            message_id = animation_states[chat_id]['message_id']
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message
            )
        except Exception as e:
            print(f"Failed to update progressive message: {e}")
    else:
        # Send new message if no animation is active
        try:
            sent_message = bot.send_message(chat_id=chat_id, text=message)
            animation_states[chat_id] = {
                'active': True,
                'message_id': sent_message.message_id
            }
        except Exception as e:
            print(f"Failed to send progressive message: {e}")

def finalize_research_feedback(update, total_tools, success_count):
    """
    Send final research completion message
    """
    if update is None:
        return
    
    # Handle different input types
    if hasattr(update, 'effective_chat'):
        chat_id = update.effective_chat.id
        bot = update.effective_chat.bot
    elif isinstance(update, int):
        chat_id = update
        bot = TELEGRAM_BOT
        if not bot:
            return
    else:
        return
    
    # Create completion message
    if success_count == total_tools:
        final_message = f"🎉 Research complete! Successfully executed {success_count} research operations."
    else:
        final_message = f"⚠️ Research completed with {success_count}/{total_tools} successful operations."
    
    # Update the animation message with final status
    if chat_id in animation_states and animation_states[chat_id].get('active'):
        try:
            message_id = animation_states[chat_id]['message_id']
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=final_message
            )
            # Mark animation as completed
            animation_states[chat_id]['active'] = False
        except Exception as e:
            print(f"Failed to finalize research message: {e}")

def clean_markdown_formatting(text: str) -> str:
    """
    Remove asterisks, double asterisks, and backticks from text for cleaner Telegram display
    """
    import re
    
    # Remove markdown formatting
    # Remove bold (**text**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove italic (*text*)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove code blocks (```text```)
    text = re.sub(r'```[^\n]*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
    # Remove inline code (`text`)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    return text

def format_deepseek_thinking(text: str) -> str:
    """
    Format Deepseek's <think>...</think> tags nicely for Telegram display
    """
    import re
    
    # Pattern to match <think>...</think> tags
    think_pattern = r'<think>(.*?)</think>'
    
    def replace_think_tags(match):
        thinking_content = match.group(1).strip()
        # Clean markdown from thinking content
        thinking_content = clean_markdown_formatting(thinking_content)
        # Format as a nice section with emoji (no markdown)
        return f"\n🤔 Thinking Process:\n{thinking_content}\n"
    
    # Replace all <think>...</think> tags with formatted versions
    formatted_text = re.sub(think_pattern, replace_think_tags, text, flags=re.DOTALL)
    
    # Clean remaining markdown from the rest of the text
    formatted_text = clean_markdown_formatting(formatted_text)
    
    return formatted_text



def send_interactive_hints(update, response: str, tools_used: list = None):
    """
    Send three useful buttons for simple research navigation.
    """
    user_id = update.effective_user.id
    
    # Create three useful buttons
    keyboard = [
        [InlineKeyboardButton("📁 Browse More Topics", callback_data=f"simple_browse_{user_id}")],
        [InlineKeyboardButton("📚 Saved Papers", callback_data=f"simple_saved_{user_id}")],
        [InlineKeyboardButton("🔍 Search Different Topic", callback_data=f"simple_search_{user_id}")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Simple message
    message_text = "💡 What would you like to do next?"
    
    # Send message with simple navigation buttons
    send_telegram_message(
        update, 
        message_text, 
        reply_markup=reply_markup
    )

def generate_button_labels_from_hints(hints: list) -> list:
    """
    Generate button labels from hint questions using LLM.
    """
    if not hints:
        return []
    
    try:
        client = Groq()
        
        prompt = """Generate concise button labels (max 20 chars each) for these follow-up questions:

{}

Format as a list, one label per line. Make each label actionable and specific to the question content. Include relevant emojis."""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a UX expert that creates concise, actionable button labels for research questions."},
                {"role": "user", "content": prompt.format('\n'.join(hints))}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        response = completion.choices[0].message.content
        if not response:
            return generate_fallback_button_labels(hints)
        
        # Parse labels from response
        labels = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line:
                # Remove any numbering or bullets
                line = line.lstrip('1234567890.- ')
                labels.append(line[:25])  # Limit length
        
        return labels[:len(hints)]  # Match number of hints
        
    except Exception as e:
        print(f"Error generating button labels: {e}")
        return generate_fallback_button_labels(hints)

def generate_fallback_button_labels(hints: list) -> list:
    """
    Fallback button labels when LLM generation fails.
    """
    fallback_labels = [
        "🔍 Explore More",
        "📚 Deep Dive", 
        "🔬 Research",
        "💡 Learn More",
        "🚀 Next Steps"
    ]
    
    labels = []
    for i, hint in enumerate(hints):
        if i < len(fallback_labels):
            labels.append(fallback_labels[i])
        else:
            labels.append(f"🔍 Option {i+1}")
    
    return labels

def generate_onboarding_research_terms():
    """
    Generate truly random and varied trending research terms for new users to explore.
    """
    import random
    import time
    
    try:
        client = Groq()
        
        # Random prompt variations for more variety
        prompt_styles = [
            "Generate 4-5 completely random and diverse research topic names from ANY field of science, technology, or academia. Be creative and unexpected!",
            "Suggest 4-5 fascinating research area names that most people haven't heard of. Think interdisciplinary, emerging, or niche fields.",
            "Create 4-5 cutting-edge topic names that would surprise and intrigue curious minds.",
            "Generate 4-5 research topic names from completely different domains - mix hard sciences, social sciences, technology, and humanities.",
            "Suggest 4-5 unconventional research area names that are gaining momentum but aren't mainstream yet."
        ]
        
        # Random research domains to inspire variety
        domain_hints = [
            "neuroscience, archaeology, materials science, linguistics",
            "marine biology, urban planning, cryptography, psychology", 
            "astrophysics, anthropology, robotics, economics",
            "genetics, philosophy, nanotechnology, sociology",
            "ecology, computer vision, history, chemistry",
            "geology, artificial intelligence, literature, medicine"
        ]
        
        # Randomly select prompt style and domain hint
        selected_prompt = random.choice(prompt_styles)
        selected_domains = random.choice(domain_hints)
        
        # Add timestamp for uniqueness
        timestamp_seed = int(time.time()) % 1000
        
        full_prompt = f"""{selected_prompt}

Consider diverse areas like: {selected_domains}

Format as simple topic names, one per line.
Examples: "Quantum Archaeology", "Urban Mycology", "Computational Linguistics"
Make each topic name specific and intriguing.
Avoid common topics like basic AI or climate change.

Seed: {timestamp_seed}"""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a creative research curator who discovers unique and surprising research topics from all fields of knowledge. Be unpredictable and diverse."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=250,
            temperature=1.0  # Maximum creativity
        )
        
        response = completion.choices[0].message.content
        if not response:
            return []
        
        # Parse topic names from response
        topics = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove any numbering, bullets, or quotes
            line = line.lstrip('1234567890.- *#"').rstrip('"')
            if line and len(line) > 3:  # Only include substantial topic names
                topics.append(line)
        
        return topics[:5]  # Limit to 5 topics
        
    except Exception as e:
        print(f"Error generating onboarding terms: {e}")
        # Random fallback topic names instead of static ones
        import random
        fallback_pools = [
            ["Quantum Computing", "AI Healthcare", "Renewable Energy", "Climate Machine Learning"],
            ["Marine Biology", "Nanotechnology Materials", "Space Archaeology", "Urban Psychology"],
            ["Cryptography Privacy", "Ancient Language AI", "Black Hole Physics", "Biotechnology Food"],
            ["Sleep Research", "VR Therapy", "Sustainable Architecture", "Social Network Psychology"],
            ["Fossil Dating", "Music AI Composition", "Earthquake Prediction", "Microbiome Mental Health"]
        ]
        return random.choice(fallback_pools)

def generate_onboarding_button_labels(questions: list) -> list:
    """
    Generate button labels for onboarding research topics using LLM.
    """
    try:
        client = Groq()
        
        prompt = """Create button labels for these research questions. Return ONLY the labels, one per line, no explanations:

{}

Requirements:
- Max 30 characters each
- Include relevant emoji
- Be specific and actionable
- No introductory text or explanations"""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You create concise button labels. Return ONLY the labels, one per line. No explanations, no introductory text, no formatting."},
                {"role": "user", "content": prompt.format('\n'.join(questions))}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        response = completion.choices[0].message.content
        if not response:
            return generate_onboarding_fallback_labels(questions)
        
        # Parse labels from response with better filtering
        labels = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.lower().startswith(('here', 'the', 'labels', 'button')):
                # Remove any numbering, bullets, or markdown formatting
                line = line.lstrip('1234567890.- *#')
                line = line.replace('**', '').replace('*', '')
                if len(line) > 3:  # Only include substantial labels
                    labels.append(line[:35])  # Limit length for mobile
        
        # Ensure we have labels for all questions
        while len(labels) < len(questions):
            labels.append(f"🔬 Research Topic {len(labels)+1}")
        
        return labels[:len(questions)]  # Match number of questions
        
    except Exception as e:
        print(f"Error generating onboarding button labels: {e}")
        return generate_onboarding_fallback_labels(questions)

def generate_onboarding_fallback_labels(questions: list) -> list:
    """
    Fallback button labels for onboarding when LLM generation fails.
    """
    fallback_labels = [
        "🤖 AI Research",
        "⚛️ Quantum Tech", 
        "🧬 Biotech",
        "🌍 Climate Science",
        "🚀 Space Tech",
        "💡 Innovation",
        "🔬 Science"
    ]
    
    labels = []
    for i in range(len(questions)):
        if i < len(fallback_labels):
            labels.append(fallback_labels[i])
        else:
            labels.append(f"🔬 Topic {i+1}")
    
    return labels

def send_onboarding_research_suggestions(update):
    """
    Send trending research topics as clickable buttons for new users.
    """
    # Generate trending research questions
    research_questions = generate_onboarding_research_terms()
    
    if not research_questions:
        send_telegram_message(update, "Welcome! Start by asking me any research question.")
        return
    
    keyboard = []
    user_id = update.effective_user.id
    
    # Generate button labels using LLM
    button_labels = generate_onboarding_button_labels(research_questions)
    
    for i, (question, button_text) in enumerate(zip(research_questions, button_labels)):
        callback_data = f"onboard_{i}_{user_id}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send message with trending research topics
    send_telegram_message(
        update,
        "🔥 Trending Research Topics - Click to explore:",
        reply_markup=reply_markup
    )
    
    # Store onboarding questions in user data for callback handling
    udata = get_user_data(user_id)
    udata['onboarding_questions'] = research_questions

def handle_hint_callback(update, context):
    """
    Handle callback when user clicks on a hint button, including smart step suggestions.
    Sends the question and triggers a bot response with enhanced tool context.
    """
    query = update.callback_query
    query.answer()  # Acknowledge the callback
    
    # Parse callback data: hint_{index}_{user_id}, onboard_{index}_{user_id}, smart_step_{index}_{user_id}
    callback_data = query.data
    
    # Handle separator buttons (do nothing)
    if callback_data == "separator":
        return
    
    if callback_data.startswith('simple_'):
        # Handle simple navigation buttons
        parts = callback_data.split('_')
        if len(parts) >= 3:
            action = parts[1]
            user_id = int(parts[2])
            
            if action == 'browse':
                # Browse More Topics - show available research folders
                try:
                    folders = get_available_folders()
                    if folders:
                        response = "📁 **Available Research Topics:**\n\n"
                        for i, folder in enumerate(folders[:10], 1):  # Limit to 10
                            topic_display = folder.replace('_', ' ').title()
                            response += f"{i}. {topic_display}\n"
                        response += "\n💡 Click any research topic button to explore papers in that area."
                    else:
                        response = "❌ No research topics found. Try searching for papers first."
                    query.message.reply_text(response, parse_mode='Markdown')
                except Exception as e:
                    query.message.reply_text(f"❌ Error loading topics: {str(e)}")
                    
            elif action == 'saved':
                # Saved Papers - show available topics with paper counts
                try:
                    folders = get_available_folders()
                    if folders:
                        response = "📚 **Saved Papers by Topic:**\n\n"
                        for i, folder in enumerate(folders[:10], 1):
                            topic_display = folder.replace('_', ' ').title()
                            # Get paper count for this topic
                            try:
                                papers = get_topic_papers(folder)
                                paper_count = len(papers) if papers else 0
                                response += f"{i}. {topic_display} ({paper_count} papers)\n"
                            except:
                                response += f"{i}. {topic_display}\n"
                        response += "\n💡 Use `/papers <topic>` to see papers in any topic."
                    else:
                        response = "❌ No saved papers found. Search for papers first to build your collection."
                    query.message.reply_text(response, parse_mode='Markdown')
                except Exception as e:
                    query.message.reply_text(f"❌ Error loading saved papers: {str(e)}")
                    
            elif action == 'search':
                # Search Different Topic - simple prompt
                prompt = "🔍 **Search for Papers**\n\nType `/search <topic>` to find papers on any research topic.\n\nExample: `/search machine learning`"
                query.message.reply_text(prompt, parse_mode='Markdown')
        
    elif callback_data.startswith('hint_') or callback_data.startswith('onboard_'):
        parts = callback_data.split('_')
        if len(parts) >= 3:
            hint_index = int(parts[1])
            user_id = int(parts[2])
            
            # Get user's stored hints or onboarding questions
            udata = get_user_data(user_id)
            
            if callback_data.startswith('onboard_'):
                # Handle onboarding research topic selection
                questions = udata.get('onboarding_questions', [])
            else:
                # Handle regular conversation hints
                questions = udata.get('current_hints', [])
            
            if hint_index < len(questions):
                question = questions[hint_index]
                
                # Extract topic from the research question for direct search
                # Remove common question words and extract the core topic
                clean_question = question.split(' ', 1)[1] if ' ' in question else question
                
                # Extract topic keywords from the question
                # Remove question words like "What are", "How does", etc.
                topic_words = clean_question.lower()
                for phrase in ["what are the latest developments in", "what are recent advances in", 
                              "how does", "what is", "explain", "research on", "studies about"]:
                    topic_words = topic_words.replace(phrase, "").strip()
                
                # Clean up the topic
                topic = topic_words.replace("?", "").replace("research", "").strip()
                if not topic:
                    topic = "artificial intelligence"  # fallback
                
                # Execute direct search command instead of AI conversation
                try:
                    # Call search_papers to get paper IDs
                    paper_ids = search_papers(topic, 10)  # Get 10 papers
                    
                    if paper_ids:
                        response = f"📚 **Recent papers on {topic.title()}:**\n\n"
                        
                        # Get details for each paper using extract_info
                        for i, paper_id in enumerate(paper_ids[:10], 1):
                            try:
                                paper_info = extract_info(paper_id)
                                if isinstance(paper_info, dict) and 'error' not in paper_info:
                                    # Successfully got paper info
                                    title = paper_info.get('title', 'Unknown Title')[:80] + ('...' if len(paper_info.get('title', '')) > 80 else '')
                                    authors = ', '.join(paper_info.get('authors', [])[:2])  # First 2 authors
                                    if len(paper_info.get('authors', [])) > 2:
                                        authors += ' et al.'
                                    
                                    published = paper_info.get('published', 'Unknown')
                                    pdf_url = paper_info.get('pdf_url', '')
                                    
                                    response += f"{i}. **{title}**\n"
                                    response += f"   Authors: {authors} ({published})\n"
                                    if pdf_url:
                                        response += f"   [📄 PDF]({pdf_url})\n"
                                    response += f"\n"
                                else:
                                    # Fallback if extract_info fails
                                    response += f"{i}. Paper ID: `{paper_id}`\n\n"
                            except Exception:
                                # Fallback if extract_info fails
                                response += f"{i}. Paper ID: `{paper_id}`\n\n"
                        
                        # No additional instruction needed - all info is shown
                    else:
                        response = f"❌ No papers found for '{topic}'. Try a different research area."
                    
                    query.message.reply_text(response, parse_mode='Markdown')
                    
                except Exception as e:
                    query.message.reply_text(f"❌ Error searching for papers on '{topic}': {str(e)}")
                    llama_command(fake_update, fake_context)

def get_deepseek_reply(messages: list, enable_tools: bool = True, update=None) -> str:
    """
    Enhanced Deepseek reply function with function calling support.
    Animation is triggered when tools are actually called.
    """
    try:
        client = Groq()
        
        # Prepare the API call parameters
        api_params = {
            "model": "deepseek-r1-distill-llama-70b",
            "messages": messages
        }
        
        # Add tools if enabled
        if enable_tools:
            api_params["tools"] = MCP_TOOLS
            api_params["tool_choice"] = "auto"
        
        completion = client.chat.completions.create(**api_params)
        message = completion.choices[0].message
        
        # Check if the model wants to call functions
        if message.tool_calls:
            total_tools = len(message.tool_calls)
            print(f"Deepseek wants to call {total_tools} function(s)")
            
            # Add the assistant's message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in message.tool_calls]
            })
            
            # Execute each tool call with progressive feedback
            success_count = 0
            for i, tool_call in enumerate(message.tool_calls, 1):
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                # Send starting status
                if update:
                    details = f"({i}/{total_tools})"
                    if arguments.get('topic'):
                        details += f" for '{arguments['topic']}'"
                    elif arguments.get('paper_id'):
                        details += f" for paper {arguments['paper_id']}"
                    send_progressive_research_update(update, function_name, 'starting', details)
                
                # Execute the function
                function_result = execute_function_call(function_name, arguments)
                
                # Handle case where function_result is None
                if function_result is None:
                    function_result = {
                        "success": False,
                        "function": function_name,
                        "error": "Function call returned None"
                    }
                
                # Send completion status
                if update:
                    if function_result.get('success'):
                        success_count += 1
                        result_details = function_result.get('summary', 'completed successfully')
                        send_progressive_research_update(update, function_name, 'completed', result_details)
                    else:
                        error_msg = function_result.get('error', 'unknown error')
                        send_progressive_research_update(update, function_name, 'completed', f"Error: {error_msg}")
                
                # Add the function result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_result)
                })
            
            # Get the final response from the model with function results
            final_completion = client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=messages
            )
            
            # Send final completion message
            if update:
                finalize_research_feedback(update, total_tools, success_count)
            
            final_response = final_completion.choices[0].message.content
            
            # Validate final response
            if not final_response or not final_response.strip():
                return "I found some research results but had trouble formatting the response. Please try asking your question again."
            
            # Format thinking tags nicely
            return format_deepseek_thinking(final_response)
        
        # No function calls, return regular response
        response = message.content
        
        # Validate regular response
        if not response or not response.strip():
            return "I'm having trouble generating a response right now. Please try asking your question again."
            
        # Format thinking tags nicely
        return format_deepseek_thinking(response)
        
    except Exception as e:
        error_str = str(e)
        print(f"Error in get_deepseek_reply: {error_str}")
        
        # Handle function calling errors
        if "tool_use_failed" in error_str or "Failed to call a function" in error_str:
            return "I tried to search for research information but encountered a technical issue with the function calling system. Please try rephrasing your question or ask me something else."
        
        # Handle token limit errors
        if "413" in error_str and "Request too large" in error_str:
            return "⚠️ Your conversation history is too long for the model's token limit. Please use /reset to start a new conversation, or ask a shorter question."
        
        return f"⚠️ Error from Groq API: {error_str}"

# ============================================================================
# TELEGRAM BOT FUNCTIONALITY
# ============================================================================

def get_user_data(user_id):
    if user_id not in user_data:
        user_data[user_id] = {
            'llama_history': [],
            'deepseek_history': [],
            'last_model': None,
            'auto_research': True  # Default to enabled
        }
    # Ensure auto_research setting exists for existing users
    if 'auto_research' not in user_data[user_id]:
        user_data[user_id]['auto_research'] = True
    return user_data[user_id]

def truncate_conversation(messages, max_tokens=MAX_TOKENS):
    """
    Automatically truncate conversation history to stay within token limits.
    Uses a simple heuristic of ~4 chars per token for estimation.
    Handles function calling messages with None content.
    """
    
    if not messages:
        return messages
    
    def safe_content_length(msg):
        """Safely get content length, handling None values from function calls"""
        content = msg.get('content')
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content)
        # Handle other content types (lists, etc.)
        return len(str(content))
    
    # Estimate tokens (rough heuristic: ~4 characters per token)
    total_chars = sum(safe_content_length(msg) for msg in messages)
    estimated_tokens = total_chars // 4
    
    if estimated_tokens <= max_tokens:
        return messages
    
    # Keep system message if present, and truncate from the beginning
    truncated = []
    if messages and messages[0].get('role') == 'system':
        truncated.append(messages[0])
        messages = messages[1:]
    
    # Keep the most recent messages that fit within the token limit
    chars_count = 0
    for msg in reversed(messages):
        msg_chars = safe_content_length(msg)
        if (chars_count + msg_chars) // 4 > max_tokens:
            break
        truncated.insert(-1 if truncated and truncated[0].get('role') == 'system' else 0, msg)
        chars_count += msg_chars
    
    return truncated

def send_telegram_message(update, text, reply_markup=None):
    """Split long messages into smaller chunks to avoid Telegram's 4096 character limit"""
    # Handle empty, None, or invalid text
    if not text or not isinstance(text, str) or not text.strip():
        error_msg = "⚠️ Sorry, I encountered an issue generating a response. Please try again."
        update.message.reply_text(error_msg, reply_markup=reply_markup)
        return
    
    # Ensure text is a string and strip whitespace
    text = str(text).strip()
    
    max_length = 4000  # Leave some buffer for safety
    
    if len(text) <= max_length:
        update.message.reply_text(text, reply_markup=reply_markup)
        return
    
    # Split message into chunks
    chunks = []
    current_chunk = ""
    
    for line in text.split('\n'):
        if len(current_chunk + line + '\n') > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                # Single line is too long, split it
                chunks.append(line[:max_length])
                current_chunk = line[max_length:] + '\n'
        else:
            current_chunk += line + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Send all chunks except the last one without reply_markup
    for chunk in chunks[:-1]:
        update.message.reply_text(chunk)
    
    # Send the last chunk with reply_markup if provided
    update.message.reply_text(chunks[-1], reply_markup=reply_markup)

def start(update, context):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "there"
    udata = get_user_data(user_id)
    
    # Initialize conversation mode to 'none' for new/returning users
    set_conversation_mode(user_id, 'none')
    
    # Create keyboard with chat options and research toggle
    reply_markup = update_keyboard(user_id)
    
    send_telegram_message(
        update,
        f"Hi {user_name}! 🚀 Welcome to Inquisita Spark - your AI-powered research companion!\n\n" +
        f"{get_conversation_status(user_id)}\n\n" +
        "🎆 What makes this special?\n" +
        "I can chat naturally AND automatically search thousands of academic papers from ArXiv when you need research insights. No more switching between tools!\n\n" +
        "🤖 Two brilliant AI assistants available:\n" +
        "• LLAMA - Great for general research and explanations\n" +
        "• Deepseek - Excellent for deep technical analysis\n\n" +
        "💬 Smart Chat Commands:\n" +
        "/llama <question> - Chat with LLAMA AI\n" +
        "/deepseek <question> - Chat with Deepseek AI\n" +
        "Just ask naturally! AI searches papers automatically when needed.\n\n" +
        "📚 Manual Research Commands:\n" +
        "/search <topic> - Search ArXiv papers directly\n" +
        "/papers <topic> - View papers for specific topic\n" +
        "/paper <id> - Get detailed paper information\n" +
        "/topics - List all available research topics\n" +
        "/prompt <topic> - Generate comprehensive research prompt\n\n" +
        "🔧 Utility Commands:\n" +
        "/reset - Clear conversation history\n" +
        "/help - Show detailed help and toggle auto-research\n\n" +
        "👇 Choose your AI assistant below to get started!",
        reply_markup=reply_markup
    )

def help_command(update, context):
    send_telegram_message(update,
        "🤖 Inquisita Spark Research Assistant Help\n\n" +
        "🧠 Smart Chat (Recommended):\n" +
        "/llama <question> - Chat with LLAMA\n" +
        "/deepseek <question> - Chat with Deepseek\n" +
        "💡 Just ask naturally! AI will search papers automatically when auto-research is enabled.\n\n" +
        "📚 Manual Research Commands:\n" +
        "/search <topic> - Search ArXiv papers directly\n" +
        "/papers <topic> - View papers for specific topic\n" +
        "/paper <id> - Get detailed paper information\n" +
        "/topics - List all available research topics\n" +
        "/prompt <topic> - Generate comprehensive research prompt\n\n" +
        "🔧 Utility Commands:\n" +
        "/reset - Clear conversation history\n" +
        "/help - Show this help message"
    )

def toggle_research_callback(update, context):
    """Handle the research toggle button callback"""
    query = update.callback_query
    user_id = query.from_user.id
    udata = get_user_data(user_id)
    
    # Toggle the setting
    udata['auto_research'] = not udata.get('auto_research', True)
    new_status = "ON" if udata['auto_research'] else "OFF"
    
    # Update the button
    keyboard = [[
        InlineKeyboardButton(
            f"🔬 Auto-Research: {new_status} (Click to toggle)",
            callback_data="toggle_research"
        )
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Answer the callback query and update the message
    query.answer(f"Auto-research {new_status}")
    query.edit_message_reply_markup(reply_markup=reply_markup)
    
    # Send confirmation message
    send_telegram_message(query, f"🔬 Auto-research is now {new_status}. This affects how LLAMA and Deepseek respond to research-related questions.")

def search_command(update, context):
    user_id = update.effective_user.id
    args = context.args
    if not args:
        send_telegram_message(update, "Usage: /search <topic>")
        return
    topic = " ".join(args)
    try:
        paper_ids = search_papers(topic)
        if not paper_ids:
            send_telegram_message(update, f"No papers found for topic '{topic}'.")
            return
        msg = f"Found {len(paper_ids)} papers for topic '{topic}':\n" + "\n".join(paper_ids)
        send_telegram_message(update, msg)
    except Exception as e:
        send_telegram_message(update, f"Error searching papers: {e}")

def papers_command(update, context):
    args = context.args
    if not args:
        send_telegram_message(update, "Usage: /papers <topic>")
        return
    topic = " ".join(args)
    try:
        papers_data = get_topic_papers(topic)
        if not papers_data:
            send_telegram_message(update, f"No papers found for topic '{topic}'.")
            return
        
        response = f"📚 Papers on '{topic}' ({len(papers_data)} found):\n\n"
        for i, paper in enumerate(papers_data, 1):
            response += f"{i}. **{paper.get('title', 'Unknown Title')}**\n"
            response += f"   ID: {paper.get('id', 'Unknown ID')}\n"
            response += f"   Authors: {', '.join(paper.get('authors', [])[:2])}{'...' if len(paper.get('authors', [])) > 2 else ''}\n"
            response += f"   Published: {paper.get('published', 'Unknown')[:10]}\n\n"
        response += "Use `/paper <id>` to get detailed information about a specific paper."
        send_telegram_message(update, response)
    except Exception as e:
        send_telegram_message(update, f"Error retrieving papers: {e}")

def paper_command(update, context):
    args = context.args
    if not args:
        send_telegram_message(update, "Usage: /paper <paper_id>")
        return
    paper_id = args[0]
    try:
        info = extract_info(paper_id)
        if "error" in info:
            send_telegram_message(update, f"Error: {info['error']}")
            return
        msg = f"Paper Info for {paper_id}:\n"
        msg += f"Title: {info.get('title', 'N/A')}\n"
        msg += f"Authors: {', '.join(info.get('authors', []))}\n"
        msg += f"Abstract: {info.get('summary', 'N/A')}\n"
        msg += f"URL: {info.get('pdf_url', 'N/A')}\n"
        send_telegram_message(update, msg)
    except Exception as e:
        send_telegram_message(update, f"Error retrieving paper info: {e}")

def topics_command(update, context):
    try:
        topics = get_available_folders()
        if not topics:
            send_telegram_message(update, "📭 No topics available yet.\n\nStart by asking me to search for papers on topics you're interested in!")
            return
        
        # Format topic names nicely (convert underscores to spaces and title case)
        formatted_topics = []
        for topic in topics:
            # Convert underscores to spaces and use title case
            formatted_name = topic.replace("_", " ").title()
            formatted_topics.append(f"📚 {formatted_name}")
        
        msg = f"📖 **Available Research Topics** ({len(topics)} total):\n\n" + "\n".join(formatted_topics)
        msg += "\n\n💡 *Use `/papers <topic>` to view papers for any topic*"
        send_telegram_message(update, msg)
    except Exception as e:
        print(f"Error in topics_command: {str(e)}")
        send_telegram_message(update, f"❌ Error retrieving topics: {str(e)}")

def llama_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    
    # Set conversation mode to llama when user uses direct command
    set_conversation_mode(user_id, 'llama')
    
    # Initialize history if not present
    if 'llama_history' not in udata:
        udata['llama_history'] = []
    
    # Get the query from message
    if not context.args:
        reply_markup = update_keyboard(user_id)
        send_telegram_message(
            update, 
            f"🤖 LLama Research Assistant\n\n{get_conversation_status(user_id)}\n\n"
            f"I can help you with research questions and general chat. I have access to ArXiv papers and can search for academic research automatically when needed.\n\n"
            f"Just ask me anything! You can continue chatting by typing messages directly.",
            reply_markup=reply_markup
        )
        
        # Show trending research topics as clickable buttons for users who just selected LLama
        send_onboarding_research_suggestions(update)
        return
        
    q = ' '.join(context.args)
    print(f"LLAMA query from user {user_id}: {q}")
    
    # Add user message to history
    udata['llama_history'].append({"role": "user", "content": q})
    
    # Truncate conversation if needed before sending to API
    udata['llama_history'] = truncate_conversation(udata['llama_history'])
    
    # Check if auto-research is enabled for this user
    auto_research_enabled = udata.get('auto_research', True)
    
    # Determine the appropriate system message based on research mode
    if auto_research_enabled:
        system_content = "You are a helpful research assistant with access to ArXiv academic papers. When users ask about research topics, recent papers, or want to find academic information, automatically use the available tools to search for and retrieve relevant papers. Integrate the research results naturally into your responses. Be conversational and helpful."
    else:
        system_content = "You are a helpful, friendly AI assistant. Engage in natural conversation and provide helpful responses on a wide variety of topics. Be conversational, informative, and engaging. Do not mention research papers, academic sources, or offer to search for papers."
    
    # Update or add system message based on current research mode
    if udata['llama_history'] and udata['llama_history'][0].get('role') == 'system':
        # Update existing system message if research mode context has changed
        udata['llama_history'][0]['content'] = system_content
    else:
        # Add new system message if none exists
        system_message = {
            "role": "system",
            "content": system_content
        }
        udata['llama_history'].insert(0, system_message)
    
    # Get reply from LLAMA with function calling enabled/disabled based on user setting
    # Animation will be handled inside get_llama_reply when tools are actually called
    reply = get_llama_reply(udata['llama_history'], enable_tools=auto_research_enabled, update=update)
    
    # Only add assistant message to history if it's not an error
    if not reply.startswith("⚠️"):
        udata['llama_history'].append({"role": "assistant", "content": reply})
    
    # Send the main response
    send_telegram_message(update, reply)
    
    # Send LLM-powered follow-up hints as separate messages
    if not reply.startswith("⚠️"):  # Only send hints for successful responses
        send_interactive_hints(update, reply)

def deepseek_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    
    # Set conversation mode to deepseek when user uses direct command
    set_conversation_mode(user_id, 'deepseek')
    
    # Initialize history if not present
    if 'deepseek_history' not in udata:
        udata['deepseek_history'] = []
    
    # Get the query from message
    if not context.args:
        reply_markup = update_keyboard(user_id)
        send_telegram_message(
            update, 
            f"🧠 Deepseek Research Assistant\n\n{get_conversation_status(user_id)}\n\n"
            f"I can help you with research questions and general chat. I have access to ArXiv papers and can search for academic research automatically when needed.\n\n"
            f"Just ask me anything! You can continue chatting by typing messages directly.",
            reply_markup=reply_markup
        )
        
        # Show trending research topics as clickable buttons for users who just selected Deepseek
        send_onboarding_research_suggestions(update)
        return
        
    q = ' '.join(context.args)
    print(f"Deepseek query from user {user_id}: {q}")
    
    # Add user message to history
    udata['deepseek_history'].append({"role": "user", "content": q})
    
    # Truncate conversation if needed before sending to API
    udata['deepseek_history'] = truncate_conversation(udata['deepseek_history'])
    
    # Check if auto-research is enabled for this user
    auto_research_enabled = udata.get('auto_research', True)
    
    # Determine the appropriate system message based on research mode
    if auto_research_enabled:
        system_content = "You are a helpful research assistant with access to ArXiv academic papers. When users ask about research topics, recent papers, or want to find academic information, automatically use the available tools to search for and retrieve relevant papers. Integrate the research results naturally into your responses. Be conversational and helpful."
    else:
        system_content = "You are a helpful, friendly AI assistant. Engage in natural conversation and provide helpful responses on a wide variety of topics. Be conversational, informative, and engaging. Do not mention research papers, academic sources, or offer to search for papers."
    
    # Update or add system message based on current research mode
    if udata['deepseek_history'] and udata['deepseek_history'][0].get('role') == 'system':
        # Update existing system message if research mode context has changed
        udata['deepseek_history'][0]['content'] = system_content
    else:
        # Add new system message if none exists
        system_message = {
            "role": "system",
            "content": system_content
        }
        udata['deepseek_history'].insert(0, system_message)
    
    # Get reply from Deepseek with function calling enabled/disabled based on user setting
    # Animation will be handled inside get_deepseek_reply when tools are actually called
    reply = get_deepseek_reply(udata['deepseek_history'], enable_tools=auto_research_enabled, update=update)
    
    # Only add assistant message to history if it's not an error
    if not reply.startswith("⚠️"):
        udata['deepseek_history'].append({"role": "assistant", "content": reply})
    
    # Send the main response
    send_telegram_message(update, reply)
    
    # Send LLM-powered follow-up hints as separate messages
    if not reply.startswith("⚠️"):  # Only send hints for successful responses
        send_interactive_hints(update, reply)

def prompt_command(update, context):
    """Generate a comprehensive research prompt for a topic"""
    if not context.args:
        send_telegram_message(update, "📝 Please provide a research topic.\n\nUsage: /prompt <topic>\nExample: /prompt quantum computing")
        return
    
    topic = " ".join(context.args)
    
    try:
        # Generate research prompt
        prompt = get_research_prompt(topic, 10)
        
        response = f"📝 **Research Prompt for '{topic}'**\n\n{prompt}"
        send_telegram_message(update, response)
        
    except Exception as e:
        print(f"Error in prompt command: {e}")
        send_telegram_message(update, "❌ Sorry, there was an error generating the research prompt. Please try again.")

def reset_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    
    # Clear conversation histories
    udata.pop('llama_history', None)
    udata.pop('deepseek_history', None)
    
    # Reset conversation mode to 'none'
    set_conversation_mode(user_id, 'none')
    
    # Update keyboard and send confirmation
    reply_markup = update_keyboard(user_id)
    send_telegram_message(
        update, 
        f"✅ **Chat History Reset**\n\n{get_conversation_status(user_id)}\n\n"
        f"Your conversation history with both LLama and Deepseek has been cleared. "
        f"Choose an AI assistant below to start a fresh conversation!",
        reply_markup=reply_markup
    )

def set_conversation_mode(user_id, mode):
    """Set explicit conversation mode: 'llama', 'deepseek', or 'none'"""
    udata = get_user_data(user_id)
    udata['conversation_mode'] = mode
    if mode != 'none':
        udata['last_model'] = mode
    print(f"User {user_id} conversation mode set to: {mode}")

def get_conversation_status(user_id):
    """Get clear status message for user"""
    udata = get_user_data(user_id)
    mode = udata.get('conversation_mode', 'none')
    research = "ON" if udata.get('auto_research', True) else "OFF"
    
    if mode == 'none':
        return "💬 Ready to chat! Choose an AI assistant to start a conversation."
    else:
        return f"💬 Chatting with {mode.upper()} | Research: {research}"

def is_in_conversation_mode(user_id):
    """Check if user is currently in an active conversation mode"""
    udata = get_user_data(user_id)
    mode = udata.get('conversation_mode', 'none')
    return mode in ['llama', 'deepseek']

def update_keyboard(user_id):
    """Update the custom keyboard with current research setting"""
    udata = get_user_data(user_id)
    auto_research_status = "ON" if udata.get('auto_research', True) else "OFF"
    
    keyboard = [
        [KeyboardButton("Chat with LLAMA"), KeyboardButton("Chat with Deepseek")],
        [KeyboardButton(f"🔬 Research: {auto_research_status}")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def message_handler(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    text = update.message.text
    
    # Handle research toggle button
    if text.startswith("🔬 Research:"):
        # Toggle the research setting
        udata['auto_research'] = not udata.get('auto_research', True)
        new_status = "ON" if udata['auto_research'] else "OFF"
        
        # Update keyboard with new status
        reply_markup = update_keyboard(user_id)
        
        send_telegram_message(
            update,
            f"🔬 Auto-research is now {new_status}. This affects how LLAMA and Deepseek respond to research-related questions.\n\n{get_conversation_status(user_id)}",
            reply_markup=reply_markup
        )
        return
    
    # Handle keyboard button presses for AI assistant selection (pure toggle)
    if text == "Chat with LLAMA":
        set_conversation_mode(user_id, 'llama')
        reply_markup = update_keyboard(user_id)
        send_telegram_message(
            update, 
            f"🤖 LLama Research Assistant\n\n{get_conversation_status(user_id)}\n\n"
            f"I can help you with research questions and general chat. I have access to ArXiv papers and can search for academic research automatically when needed.\n\n"
            f"Just ask me anything! You can continue chatting by typing messages directly.",
            reply_markup=reply_markup
        )
        # Show trending research topics as clickable buttons for users who just selected LLama
        send_onboarding_research_suggestions(update)
        return
        
    elif text == "Chat with Deepseek":
        set_conversation_mode(user_id, 'deepseek')
        reply_markup = update_keyboard(user_id)
        send_telegram_message(
            update, 
            f"🧠 Deepseek Research Assistant\n\n{get_conversation_status(user_id)}\n\n"
            f"I can help you with research questions and general chat. I have access to ArXiv papers and can search for academic research automatically when needed.\n\n"
            f"Just ask me anything! You can continue chatting by typing messages directly.",
            reply_markup=reply_markup
        )
        # Show trending research topics as clickable buttons for users who just selected Deepseek
        send_onboarding_research_suggestions(update)
        return
    
    # Check if user is in an active conversation mode
    if is_in_conversation_mode(user_id):
        # User is in conversation mode, route to appropriate AI
        mode = udata.get('conversation_mode', 'llama')
        context.args = text.split()
        
        if mode == 'deepseek':
            return deepseek_command(update, context)
        else:
            return llama_command(update, context)
    else:
        # User is not in conversation mode, guide them to choose
        reply_markup = update_keyboard(user_id)
        send_telegram_message(
            update,
            f"🤖 Choose an AI Assistant\n\n{get_conversation_status(user_id)}\n\n"
            f"Your message: {text}\n\n"
            f"Please select Chat with LLAMA or Chat with Deepseek to start a conversation, "
            f"or use specific commands like `/llama {text}` or `/deepseek {text}`.",
            reply_markup=reply_markup
        )

# Register handlers with the dispatcher
if telegram_dispatcher:
    telegram_dispatcher.add_handler(CommandHandler("start", start))
    telegram_dispatcher.add_handler(CommandHandler("help", help_command))
    telegram_dispatcher.add_handler(CommandHandler("search", search_command))
    telegram_dispatcher.add_handler(CommandHandler("papers", papers_command))
    telegram_dispatcher.add_handler(CommandHandler("paper", paper_command))
    telegram_dispatcher.add_handler(CommandHandler("topics", topics_command))
    telegram_dispatcher.add_handler(CommandHandler("prompt", prompt_command))
    telegram_dispatcher.add_handler(CommandHandler("llama", llama_command))
    telegram_dispatcher.add_handler(CommandHandler("deepseek", deepseek_command))
    telegram_dispatcher.add_handler(CommandHandler("reset", reset_command))
    # Add callback handler for interactive hint buttons
    telegram_dispatcher.add_handler(CallbackQueryHandler(handle_hint_callback))
    # Add handler for regular text messages (must be added last)
    telegram_dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, message_handler))

# ============================================================================
# FLASK WEB ROUTES
# ============================================================================

@app.route('/telegram_webhook', methods=['POST'])
def telegram_webhook():
    try:
        # Log that we received a webhook
        print("\n=== Received Telegram webhook ===")
        
        # Check if we have a valid token
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_BOT:
            error_msg = "ERROR: TELEGRAM_BOT_TOKEN environment variable not set or invalid!"
            print(error_msg)
            return jsonify(success=False, error=error_msg), 500
            
        # Log headers for debugging
        print(f"Headers: {dict(request.headers)}")
        
        # Get the JSON data
        data = request.get_json(force=True)
        print(f"Received update: {json.dumps(data, indent=2)}")
        
        if not data:
            error_msg = "ERROR: No data received in webhook"
            print(error_msg)
            return jsonify(success=False, error=error_msg), 400
            
        # Process the update
        update = Update.de_json(data, TELEGRAM_BOT)
        if update:
            print(f"Processing update ID: {update.update_id}")
            try:
                telegram_dispatcher.process_update(update)
                print(f"Successfully processed update {update.update_id}")
            except Exception as e:
                print(f"ERROR processing update {update.update_id}: {str(e)}")
                return jsonify(success=False, error=f"Error processing update: {str(e)}"), 500
        else:
            print("WARNING: Received invalid update format from Telegram")
            return jsonify(success=False, error="Invalid update format"), 400
            
        return jsonify(success=True)
    except Exception as e:
        import traceback
        error_msg = f"ERROR in telegram_webhook: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify(success=False, error=error_msg), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Handle the POST request (form submission from index.html)
        name = request.form.get('q', 'Guest')
        return render_template('main.html', name=name)
    # Handle GET request
    return render_template('main.html')

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/search', methods=['POST'])
def web_search():
    topic = request.form.get('topic')
    if not topic:
        return render_template('research.html', error="Please provide a research topic.")
    
    try:
        # First, search for papers and get their IDs
        paper_ids = search_papers(topic, max_results=10)
        
        papers_data = []
        
        if paper_ids:
            # Try to get papers from the topic first (in case they're already stored)
            papers_info = get_topic_papers(topic)
            if papers_info and isinstance(papers_info, list) and len(papers_info) > 0:
                papers_data = papers_info
                print(f"Retrieved {len(papers_data)} papers from stored topic")
            else:
                # If no stored papers, fetch details for each paper ID
                print("No stored papers found, fetching details for each paper ID")
                for paper_id in paper_ids:
                    try:
                        paper_info = extract_info(paper_id)
                        if paper_info and not paper_info.get('error'):
                            papers_data.append(paper_info)
                            print(f"Successfully fetched info for paper {paper_id}")
                        else:
                            print(f"Failed to fetch info for paper {paper_id}: {paper_info}")
                    except Exception as e:
                        print(f"Error fetching paper {paper_id}: {e}")
                        continue
        
        print(f"Final papers_data count: {len(papers_data)}")
        return render_template('search_results.html', 
                             topic=topic, 
                             papers=papers_data, 
                             paper_count=len(papers_data))
    except Exception as e:
        return render_template('research.html', error=f"Error searching papers: {str(e)}")

@app.route('/topics')
def web_topics():
    try:
        folders = get_available_folders()
        topics_data = {}
        
        for folder in folders:
            topic_name = folder.replace("_", " ").title()
            papers_info = get_topic_papers(folder.replace("_", " "))
            # get_topic_papers returns a list, not a string
            if papers_info and isinstance(papers_info, list) and len(papers_info) > 0:
                topics_data[topic_name] = len(papers_info)
        
        return render_template('topics.html', topics=topics_data)
    except Exception as e:
        return render_template('topics.html', error=f"Error loading topics: {str(e)}")

@app.route('/llama', methods=['GET', 'POST'])
def llama():
    if request.method == 'POST':
        q = request.form.get("q")
        if q:
            # Add system message for research assistant behavior
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful research assistant with access to ArXiv academic papers. When users ask about research topics, recent papers, or want to find academic information, automatically use the available tools to search for and retrieve relevant papers. Integrate the research results naturally into your responses. Be conversational and helpful."
                },
                {"role": "user", "content": q}
            ]
            # Use enhanced function calling
            reply = get_llama_reply(messages, enable_tools=True)
            return render_template("llama_reply.html", r=reply)
    return render_template("llama.html")

@app.route('/deepseek', methods=['GET', 'POST'])
def deepseek():
    if request.method == 'POST':
        q = request.form.get("q")
        if q:
            # Add system message for research assistant behavior
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful research assistant with access to ArXiv academic papers. When users ask about research topics, recent papers, or want to find academic information, automatically use the available tools to search for and retrieve relevant papers. Integrate the research results naturally into your responses. Be conversational and helpful."
                },
                {"role": "user", "content": q}
            ]
            # Use enhanced function calling
            reply = get_deepseek_reply(messages, enable_tools=True)
            return render_template("deepseek_reply.html", r=reply)
    return render_template("deepseek.html")

if __name__ == "__main__":
    app.run(debug=True)
