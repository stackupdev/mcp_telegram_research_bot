import os
import json
import asyncio
from typing import List
from flask import Flask, render_template, request, jsonify
from groq import Groq
from telegram import Update, Bot, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
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

def search_papers(topic: str, max_results: int = 5) -> List[str]:
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
        print("Attempting to read resource: papers://folders")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(read_mcp_resource("papers://folders"))
        loop.close()
        
        print(f"Raw result from MCP server: {result}")
        print(f"Result type: {type(result)}")
        
        # Handle MCP ReadResourceResult object
        text_content = None
        if hasattr(result, 'contents') and result.contents:
            # Extract text from the first content item
            first_content = result.contents[0]
            if hasattr(first_content, 'text'):
                text_content = first_content.text
                print(f"Extracted text content: {text_content}")
        elif isinstance(result, str):
            text_content = result
        
        if text_content:
            lines = text_content.split('\n')
            folders = []
            print(f"Processing {len(lines)} lines from text content")
            for i, line in enumerate(lines):
                print(f"Line {i}: '{line.strip()}'")
                if line.strip().startswith('- '):
                    folder_name = line.strip()[2:]  # Remove '- ' prefix
                    folders.append(folder_name)
                    print(f"Found folder: {folder_name}")
            print(f"Total folders found: {len(folders)}")
            return folders
        elif isinstance(result, dict):
            print(f"Result is dict with keys: {result.keys()}")
            # Handle if result is a dictionary
            return list(result.keys()) if result else []
        elif isinstance(result, list):
            print(f"Result is list with {len(result)} items")
            return result
        elif result is None:
            print("Got None result from MCP resource")
            return []
        else:
            print(f"Unexpected result type: {type(result)}")
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
        
        # Parse the markdown content to extract paper information
        if isinstance(result, str):
            if "No papers found" in result:
                return []
            
            # Try to extract paper data from the markdown
            # This is a simplified parser - you might want to enhance it
            papers = []
            lines = result.split('\n')
            current_paper = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('## ') and not line.startswith('## Papers on'):
                    # New paper title
                    if current_paper:
                        papers.append(current_paper)
                    current_paper = {'title': line[3:]}
                elif line.startswith('- **Paper ID**:'):
                    current_paper['id'] = line.split(':', 1)[1].strip()
                elif line.startswith('- **Authors**:'):
                    authors_str = line.split(':', 1)[1].strip()
                    current_paper['authors'] = [a.strip() for a in authors_str.split(',')]
                elif line.startswith('- **Published**:'):
                    current_paper['published'] = line.split(':', 1)[1].strip()
                elif line.startswith('### Summary'):
                    # Next non-empty line should be the summary
                    continue
                elif current_paper and 'summary' not in current_paper and line and not line.startswith('-'):
                    current_paper['summary'] = line.replace('...', '')
            
            if current_paper:
                papers.append(current_paper)
            
            return papers
        return []
    except Exception as e:
        print(f"Error calling get_topic_papers via MCP: {e}")
        return []

def get_research_prompt(topic: str, num_papers: int = 5) -> str:
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
                        "description": "Maximum number of papers to return (default: 5, max: 10)",
                        "default": 5
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
                        "description": "Number of papers to include in the research analysis (default: 5)",
                        "default": 5
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
            
            max_results = arguments.get("max_results", 5)
            try:
                max_results = int(max_results)
                max_results = max(1, min(max_results, 10))  # Clamp between 1-10
            except (ValueError, TypeError):
                max_results = 5
            
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
            
            num_papers = arguments.get("num_papers", 5)
            if not isinstance(num_papers, int) or num_papers < 1:
                num_papers = 5
            
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

def get_llama_reply(messages: list, enable_tools: bool = True) -> str:
    """
    Enhanced LLama reply function with function calling support.
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
            print(f"LLama wants to call {len(message.tool_calls)} function(s)")
            
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
            
            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                # Execute the function
                function_result = execute_function_call(function_name, arguments)
                
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
        
        # Handle token limit errors
        if "413" in error_str and "Request too large" in error_str:
            return "⚠️ Your conversation history is too long for the model's token limit. Please use /reset to start a new conversation, or ask a shorter question."
        
        return f"⚠️ Error from Groq API: {error_str}"

# Global dictionary to track animation states
animation_states = {}

def send_animated_search_message(update):
    """
    Send an animated search message to indicate research is in progress
    """
    import time
    import threading
    
    chat_id = update.effective_chat.id
    
    def animate_search():
        search_frames = [
            "🔍 Searching for relevant research",
            "🔍 Searching for relevant research.",
            "🔍 Searching for relevant research..",
            "🔍 Searching for relevant research...",
            "🔍 Searching for relevant research....",
            "🔍 Searching for relevant research....."
        ]
        
        try:
            # Send initial message using bot directly to get message object
            message = update.effective_chat.bot.send_message(
                chat_id=chat_id,
                text=search_frames[0]
            )
            
            # Store animation state
            animation_states[chat_id] = {
                'active': True,
                'message_id': message.message_id
            }
            
            # Cycle through the animation frames continuously until stopped
            frame_index = 1
            while animation_states.get(chat_id, {}).get('active', False):
                time.sleep(0.1)
                try:
                    # Edit the message to show animation
                    update.effective_chat.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message.message_id,
                        text=search_frames[frame_index]
                    )
                    # Cycle through frames
                    frame_index = (frame_index + 1) % len(search_frames)
                    if frame_index == 0:  # Skip the first frame (no dots) in cycling
                        frame_index = 1
                except Exception as e:
                    # If editing fails, stop animation
                    print(f"Animation edit failed: {e}")
                    break
                
        except Exception as e:
            # If animation fails completely, fall back to simple message
            print(f"Animation failed: {e}")
            send_telegram_message(update, "🔍 Searching for relevant research...")
        finally:
            # Clean up animation state
            if chat_id in animation_states:
                del animation_states[chat_id]
    
    # Run animation in a separate thread to not block
    thread = threading.Thread(target=animate_search)
    thread.daemon = True
    thread.start()
    
    return thread

def stop_search_animation(update):
    """
    Stop the animated search message for a specific chat
    """
    chat_id = update.effective_chat.id
    if chat_id in animation_states:
        animation_states[chat_id]['active'] = False

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

def get_deepseek_reply(messages: list, enable_tools: bool = True) -> str:
    """
    Enhanced Deepseek reply function with function calling support.
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
            print(f"Deepseek wants to call {len(message.tool_calls)} function(s)")
            
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
            
            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                # Execute the function
                function_result = execute_function_call(function_name, arguments)
                
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
    auto_research_status = "ON" if udata.get('auto_research', True) else "OFF"
    
    # Create keyboard with chat options and research toggle
    keyboard = [
        [KeyboardButton("Chat with LLAMA"), KeyboardButton("Chat with Deepseek")],
        [KeyboardButton(f"🔬 Research: {auto_research_status}")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    send_telegram_message(
        update,
        f"🔬 Welcome to Inquisita Spark Research Assistant, {user_name}!\n\n" +
        "🤖 Intelligent Research Chat:\n" +
        "• Chat naturally with LLAMA or Deepseek\n" +
        "• AI automatically searches ArXiv papers when needed\n" +
        "• Get research insights in conversational format\n\n" +
        "📚 Manual Research Tools:\n" +
        "• Direct paper search and topic browsing\n" +
        "• Detailed paper information retrieval\n\n" +
        "💡 Just ask questions like:\n" +
        "• \"What are the latest papers on quantum computing?\"\n" +
        "• \"Find research about neural networks\"\n" +
        "• \"Tell me about recent AI developments\"\n\n" +
        "Select a chat mode below or toggle research mode!",
        reply_markup=reply_markup
    )

def help_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    auto_research_status = "ON" if udata.get('auto_research', True) else "OFF"
    
    # Create inline keyboard for research toggle
    keyboard = [[
        InlineKeyboardButton(
            f"🔬 Auto-Research: {auto_research_status} (Click to toggle)",
            callback_data="toggle_research"
        )
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
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
        "/topics - List all available research topics\n\n" +
        "🔧 Utility Commands:\n" +
        "/reset - Clear conversation history\n" +
        "/help - Show this help message\n\n" +
        "✨ Example Questions:\n" +
        "• \"What's new in machine learning research?\"\n" +
        "• \"Find papers about quantum computing\"\n" +
        "• \"Explain recent developments in AI safety\"",
        reply_markup=reply_markup
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
    
    # Track which model was last used
    udata['last_model'] = 'llama'
    
    # Initialize history if not present
    if 'llama_history' not in udata:
        udata['llama_history'] = []
    
    # Get the query from message
    if not context.args:
        send_telegram_message(update, "🤖 **LLama Research Assistant**\n\nI can help you with research questions and general chat. I have access to ArXiv papers and can search for academic research automatically when needed.\n\nJust ask me anything!")
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
    
    # Send animated search indicator for research queries (only if auto-research is enabled)
    search_thread = None
    if auto_research_enabled and any(keyword in q.lower() for keyword in ['paper', 'research', 'study', 'recent', 'latest', 'find', 'search', 'academic']):
        search_thread = send_animated_search_message(update)
    
    # Get reply from LLAMA with function calling enabled/disabled based on user setting
    reply = get_llama_reply(udata['llama_history'], enable_tools=auto_research_enabled)
    
    # Stop the search animation if it was started
    if search_thread:
        stop_search_animation(update)
    
    # Only add assistant message to history if it's not an error
    if not reply.startswith("⚠️"):
        udata['llama_history'].append({"role": "assistant", "content": reply})
        
    send_telegram_message(update, reply)

def deepseek_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    
    # Track which model was last used
    udata['last_model'] = 'deepseek'
    
    # Initialize history if not present
    if 'deepseek_history' not in udata:
        udata['deepseek_history'] = []
    
    # Get the query from message
    if not context.args:
        send_telegram_message(update, "🧠 **Deepseek Research Assistant**\n\nI can help you with research questions and general chat. I have access to ArXiv papers and can search for academic research automatically when needed.\n\nJust ask me anything!")
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
    
    # Send animated search indicator for research queries (only if auto-research is enabled)
    search_thread = None
    if auto_research_enabled and any(keyword in q.lower() for keyword in ['paper', 'research', 'study', 'recent', 'latest', 'find', 'search', 'academic']):
        search_thread = send_animated_search_message(update)
    
    # Get reply from Deepseek with function calling enabled/disabled based on user setting
    reply = get_deepseek_reply(udata['deepseek_history'], enable_tools=auto_research_enabled)
    
    # Stop the search animation if it was started
    if search_thread:
        stop_search_animation(update)
    
    # Only add assistant message to history if it's not an error
    if not reply.startswith("⚠️"):
        udata['deepseek_history'].append({"role": "assistant", "content": reply})
        
    send_telegram_message(update, reply)

def reset_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    udata.pop('llama_history', None)
    udata.pop('deepseek_history', None)
    send_telegram_message(update, "✅ Your chat history has been reset.")

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
            f"🔬 Auto-research is now {new_status}. This affects how LLAMA and Deepseek respond to research-related questions.",
            reply_markup=reply_markup
        )
        return
    
    # Handle keyboard button presses
    if text == "Chat with LLAMA":
        context.args = ["Hi,", "I'd", "like", "to", "chat"]
        return llama_command(update, context)
    elif text == "Chat with Deepseek":
        context.args = ["Hi,", "I'd", "like", "to", "chat"]
        return deepseek_command(update, context)
    
    # Default: treat as a question for the last used model or LLAMA
    model = udata.get('last_model', 'llama')
    if model == 'deepseek':
        context.args = text.split()
        return deepseek_command(update, context)
    else:
        context.args = text.split()
        return llama_command(update, context)

# Register handlers with the dispatcher
if telegram_dispatcher:
    telegram_dispatcher.add_handler(CommandHandler("start", start))
    telegram_dispatcher.add_handler(CommandHandler("help", help_command))
    telegram_dispatcher.add_handler(CommandHandler("search", search_command))
    telegram_dispatcher.add_handler(CommandHandler("papers", papers_command))
    telegram_dispatcher.add_handler(CommandHandler("paper", paper_command))
    telegram_dispatcher.add_handler(CommandHandler("topics", topics_command))
    telegram_dispatcher.add_handler(CommandHandler("llama", llama_command))
    telegram_dispatcher.add_handler(CommandHandler("deepseek", deepseek_command))
    telegram_dispatcher.add_handler(CommandHandler("reset", reset_command))
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
        paper_ids = search_papers(topic, max_results=10)
        papers_info = get_topic_papers(topic)
        papers_data = json.loads(papers_info) if not papers_info.startswith("No papers") else {}
        
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
            if not papers_info.startswith("No papers"):
                papers_data = json.loads(papers_info)
                topics_data[topic_name] = len(papers_data)
        
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
