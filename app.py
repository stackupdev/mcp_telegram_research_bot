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

RESEARCH_SERVER_URL = "https://mcp-arxiv-server.onrender.com"

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

def search_papers(query: str, max_results: int = 10, date_from: str = None, categories: List[str] = None) -> List[dict]:
    """
    Search for papers with optional filters using the new arXiv server API.
    Returns: List of paper metadata dictionaries
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return []
    
    try:
        # Prepare arguments for the search_papers tool
        args = {
            "query": query,
            "max_results": max_results
        }
        if date_from:
            args["date_from"] = date_from
        if categories:
            args["categories"] = categories
            
        print(f"Calling search_papers tool with args: {args}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_mcp_tool("search_papers", **args))
        loop.close()
        
        print(f"Search result: {result}")
        
        # Handle MCP tool call result
        if hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                # Extract text from TextContent objects
                if hasattr(content[0], 'text'):
                    try:
                        # Try to parse as JSON
                        papers = json.loads(content[0].text)
                        return papers if isinstance(papers, list) else [papers]
                    except json.JSONDecodeError:
                        return [{"info": content[0].text}]
                else:
                    return [{"info": str(content[0])}]
            elif isinstance(content, str):
                try:
                    papers = json.loads(content)
                    return papers if isinstance(papers, list) else [papers]
                except json.JSONDecodeError:
                    return [{"info": content}]
        elif isinstance(result, list):
            return result
        elif isinstance(result, str):
            try:
                papers = json.loads(result)
                return papers if isinstance(papers, list) else [papers]
            except json.JSONDecodeError:
                return [{"info": result}]
        else:
            return []
    except Exception as e:
        print(f"Error calling search_papers via MCP: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

def download_paper(paper_id: str) -> dict:
    """
    Download a paper by its arXiv ID using the remote MCP server.
    Returns: Download result with status information
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return {"error": "MCP client not available"}
    
    try:
        print(f"Calling download_paper tool with paper_id='{paper_id}'")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_mcp_tool("download_paper", paper_id=paper_id))
        loop.close()
        
        print(f"Download result: {result}")
        
        # Handle MCP tool call result
        if hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                # Extract text from first TextContent object
                if hasattr(content[0], 'text'):
                    return {"status": "success", "message": content[0].text}
                else:
                    return {"status": "success", "message": str(content[0])}
            elif isinstance(content, str):
                return {"status": "success", "message": content}
        elif isinstance(result, str):
            return {"status": "success", "message": result}
        elif result is None:
            return {"error": "No result from download operation"}
        else:
            return {"status": "success", "message": str(result)}
    except Exception as e:
        print(f"Error calling download_paper via MCP: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Download failed: {str(e)}"}

def list_papers() -> List[dict]:
    """
    List all downloaded papers using the remote MCP server.
    Returns: List of paper information dictionaries
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return []
    
    try:
        print("Calling list_papers tool")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_mcp_tool("list_papers"))
        loop.close()
        
        print(f"List papers result: {result}")
        
        # Handle MCP tool call result
        if hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                if hasattr(content[0], 'text'):
                    try:
                        # Try to parse as JSON
                        papers = json.loads(content[0].text)
                        return papers if isinstance(papers, list) else [papers]
                    except json.JSONDecodeError:
                        return [{"info": content[0].text}]
                else:
                    return [{"info": str(content[0])}]
            elif isinstance(content, str):
                try:
                    papers = json.loads(content)
                    return papers if isinstance(papers, list) else [papers]
                except json.JSONDecodeError:
                    return [{"info": content}]
        elif isinstance(result, list):
            return result
        elif isinstance(result, str):
            try:
                papers = json.loads(result)
                return papers if isinstance(papers, list) else [papers]
            except json.JSONDecodeError:
                return [{"info": result}]
        else:
            return []
    except Exception as e:
        print(f"Error calling list_papers via MCP: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

def read_paper(paper_id: str) -> dict:
    """
    Read the content of a downloaded paper by its arXiv ID using the remote MCP server.
    Returns: Paper content and metadata
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return {"error": "MCP client not available"}
    
    try:
        print(f"Calling read_paper tool with paper_id='{paper_id}'")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_mcp_tool("read_paper", paper_id=paper_id))
        loop.close()
        
        print(f"Read paper result: {result}")
        
        # Handle MCP tool call result
        if hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                # Extract text from first TextContent object
                if hasattr(content[0], 'text'):
                    try:
                        # Try to parse as JSON if it's structured data
                        parsed = json.loads(content[0].text)
                        return parsed
                    except json.JSONDecodeError:
                        return {"content": content[0].text}
                else:
                    return {"content": str(content[0])}
            elif isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    return parsed
                except json.JSONDecodeError:
                    return {"content": content}
        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
                return parsed
            except json.JSONDecodeError:
                return {"content": result}
        elif result is None:
            return {"error": "Paper not found or not downloaded"}
        else:
            return {"content": str(result)}
    except Exception as e:
        print(f"Error calling read_paper via MCP: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Read failed: {str(e)}"}

def get_deep_paper_analysis(paper_id: str) -> str:
    """
    Get a comprehensive analysis of a paper using the deep-paper-analysis prompt.
    Returns: Detailed analysis text
    """
    if not mcp_client_factory:
        print("MCP client factory not initialized")
        return "MCP client not available"
    
    try:
        print(f"Calling deep-paper-analysis prompt with paper_id='{paper_id}'")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Use the prompt primitive instead of tool
        async def get_prompt():
            async with mcp_client_factory() as streams:
                read_stream, write_stream = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.get_prompt("deep-paper-analysis", arguments={"paper_id": paper_id})
                    return result
        
        result = loop.run_until_complete(get_prompt())
        loop.close()
        
        print(f"Deep analysis prompt result: {result}")
        
        # Handle prompt result
        if hasattr(result, 'messages') and result.messages:
            # Extract content from prompt messages
            analysis_parts = []
            for message in result.messages:
                if hasattr(message, 'content'):
                    if hasattr(message.content, 'text'):
                        analysis_parts.append(message.content.text)
                    else:
                        analysis_parts.append(str(message.content))
            return "\n\n".join(analysis_parts)
        elif hasattr(result, 'content'):
            if isinstance(result.content, str):
                return result.content
            else:
                return str(result.content)
        elif isinstance(result, str):
            return result
        else:
            return f"Analysis generated for paper {paper_id}"
    except Exception as e:
        print(f"Error calling deep-paper-analysis prompt via MCP: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return f"Error generating analysis: {str(e)}"

# ============================================================================
# FUNCTION CALLING INFRASTRUCTURE
# ============================================================================

# Define function schemas for the new arXiv MCP tools
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for academic papers on ArXiv with optional filters. Use this when the user asks about research papers, recent studies, or wants to find papers on a specific topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for papers (e.g., 'transformer architecture', 'quantum computing', 'machine learning')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to return (default: 10)",
                        "default": 10
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Filter papers from this date onwards (format: YYYY-MM-DD)"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by arXiv categories (e.g., ['cs.AI', 'cs.LG'])"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_paper",
            "description": "Download a paper by its arXiv ID for detailed analysis. Use this when you need to access the full content of a specific paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The arXiv paper ID (e.g., '2401.12345')"
                    }
                },
                "required": ["paper_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_papers",
            "description": "List all downloaded papers. Use this to see what papers are available for detailed analysis.",
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
            "name": "read_paper",
            "description": "Read the content of a downloaded paper. Use this to access the full text and metadata of a paper for detailed analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The arXiv paper ID (e.g., '2401.12345')"
                    }
                },
                "required": ["paper_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_deep_paper_analysis",
            "description": "Get a comprehensive analysis of a paper using the deep-paper-analysis prompt. Use this for in-depth academic analysis of a specific paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The arXiv paper ID (e.g., '2401.12345')"
                    }
                },
                "required": ["paper_id"]
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
            query = arguments.get("query")
            if not query or not isinstance(query, str):
                return {
                    "success": False,
                    "function": function_name,
                    "error": "Query parameter is required and must be a string"
                }
            
            max_results = arguments.get("max_results", 10)
            try:
                max_results = int(max_results)
                max_results = max(1, min(max_results, 50))  # Clamp between 1-50
            except (ValueError, TypeError):
                max_results = 10
            
            date_from = arguments.get("date_from")
            categories = arguments.get("categories")
            
            result = search_papers(query, max_results, date_from, categories)
            
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Found {len(result)} papers for query '{query}'"
            }
            
        elif function_name == "download_paper":
            paper_id = arguments.get("paper_id")
            if not paper_id or not isinstance(paper_id, str):
                return {
                    "success": False,
                    "function": function_name,
                    "error": "Paper ID parameter is required and must be a string"
                }
            
            result = download_paper(paper_id)
            if "error" in result:
                return {
                    "success": False,
                    "function": function_name,
                    "error": result["error"]
                }
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Downloaded paper {paper_id}"
            }
            
        elif function_name == "list_papers":
            result = list_papers()
            
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Found {len(result)} downloaded papers"
            }
            
        elif function_name == "read_paper":
            paper_id = arguments.get("paper_id")
            if not paper_id or not isinstance(paper_id, str):
                return {
                    "success": False,
                    "function": function_name,
                    "error": "Paper ID parameter is required and must be a string"
                }
            
            result = read_paper(paper_id)
            if "error" in result:
                return {
                    "success": False,
                    "function": function_name,
                    "error": result["error"]
                }
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Read content of paper {paper_id}"
            }
            
        elif function_name == "get_deep_paper_analysis":
            paper_id = arguments.get("paper_id")
            if not paper_id or not isinstance(paper_id, str):
                return {
                    "success": False,
                    "function": function_name,
                    "error": "Paper ID parameter is required and must be a string"
                }
            
            result = get_deep_paper_analysis(paper_id)
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "summary": f"Generated comprehensive analysis for paper {paper_id}"
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
    Enhanced LLama reply function with function calling support for the new arXiv MCP tools.
    Animation is triggered when tools are actually called.
    """
    try:
        client = Groq()
        
        # Prepare the API call parameters
        api_params = {
            "model": "llama-3.1-70b-versatile",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        # Add tools if enabled
        if enable_tools:
            api_params["tools"] = MCP_TOOLS
            api_params["tool_choice"] = "auto"
        
        # Make the API call
        response = client.chat.completions.create(**api_params)
        
        # Handle tool calls if present
        if response.choices[0].message.tool_calls:
            # Process each tool call
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"LLama is calling function: {function_name} with args: {function_args}")
                
                # Send animated search message if update is available
                if update:
                    try:
                        update.message.reply_text(f"🔍 Searching research papers using {function_name}...")
                    except:
                        pass
                
                # Execute the function call
                function_result = execute_function_call(function_name, function_args)
                
                # Add the assistant's tool call message to conversation
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                # Add the function result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_result)
                })
            
            # Make another API call to get the final response
            final_response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            return final_response.choices[0].message.content
        else:
            # No tool calls, return the regular response
            return response.choices[0].message.content
            
    except Exception as e:
        print(f"Error getting LLama reply: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

def get_deepseek_reply(messages: list, enable_tools: bool = True, update=None) -> str:
    """
    Enhanced Deepseek reply function with function calling support for the new arXiv MCP tools.
    Animation is triggered when tools are actually called.
    """
    try:
        client = Groq()
        
        # Prepare the API call parameters
        api_params = {
            "model": "deepseek-r1-distill-llama-70b",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        # Add tools if enabled
        if enable_tools:
            api_params["tools"] = MCP_TOOLS
            api_params["tool_choice"] = "auto"
        
        # Make the API call
        response = client.chat.completions.create(**api_params)
        
        # Handle tool calls if present
        if response.choices[0].message.tool_calls:
            # Process each tool call
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Deepseek is calling function: {function_name} with args: {function_args}")
                
                # Send animated search message if update is available
                if update:
                    try:
                        update.message.reply_text(f"🔍 Searching research papers using {function_name}...")
                    except:
                        pass
                
                # Execute the function call
                function_result = execute_function_call(function_name, function_args)
                
                # Add the assistant's tool call message to conversation
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                # Add the function result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_result)
                })
            
            # Make another API call to get the final response
            final_response = client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            # Format Deepseek thinking tags if present
            content = final_response.choices[0].message.content
            if content and "<think>" in content:
                content = format_deepseek_thinking(content)
            
            return content
        else:
            # No tool calls, return the regular response
            content = response.choices[0].message.content
            if content and "<think>" in content:
                content = format_deepseek_thinking(content)
            return content
            
    except Exception as e:
        print(f"Error getting Deepseek reply: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

# Global dictionary to track progressive research feedback states
animation_states = {}

def send_progressive_research_update(update, tool_name, status, details=None):
    """
    Send progressive research feedback with specific tool status updates
    """
    user_id = update.effective_user.id
    
    # Initialize animation state if not exists
    if user_id not in animation_states:
        animation_states[user_id] = {
            'active': False,
            'tools_called': [],
            'current_tool': None
        }
    
    state = animation_states[user_id]
    
    if status == "start":
        state['active'] = True
        state['current_tool'] = tool_name
        if tool_name not in state['tools_called']:
            state['tools_called'].append(tool_name)
        
        # Send tool-specific start message
        tool_messages = {
            "search_papers": "🔍 Searching arXiv papers...",
            "download_paper": "📥 Downloading paper...",
            "list_papers": "📋 Listing downloaded papers...",
            "read_paper": "📖 Reading paper content...",
            "get_deep_paper_analysis": "🧠 Generating deep analysis..."
        }
        
        message = tool_messages.get(tool_name, f"🔧 Using {tool_name}...")
        if details:
            message += f" {details}"
            
        try:
            update.message.reply_text(message)
        except:
            pass
    
    elif status == "complete":
        state['current_tool'] = None
        
        # Send completion message
        try:
            if details:
                update.message.reply_text(f"✅ {details}")
        except:
            pass

def finalize_research_feedback(update, total_tools, success_count):
    """
    Send final research completion message
    """
    user_id = update.effective_user.id
    
    if user_id in animation_states:
        animation_states[user_id]['active'] = False
        animation_states[user_id]['current_tool'] = None
    
    if total_tools > 0:
        try:
            if success_count == total_tools:
                update.message.reply_text(f"🎉 Research complete! Successfully used {success_count} tools.")
            else:
                update.message.reply_text(f"⚠️ Research complete with {success_count}/{total_tools} tools successful.")
        except:
            pass

def clean_markdown_formatting(text: str) -> str:
    """
    Remove asterisks, double asterisks, and backticks from text for cleaner Telegram display
    """
    if not text:
        return text
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
    text = re.sub(r'`(.*?)`', r'\1', text)        # Remove `code`
    text = re.sub(r'```(.*?)```', r'\1', text, flags=re.DOTALL)  # Remove ```code blocks```
    
    return text

def format_deepseek_thinking(text: str) -> str:
    """
    Format Deepseek's <think>...</think> tags nicely for Telegram display
    """
    if not text:
        return text
    
    # Replace <think> tags with emoji sections
    text = re.sub(r'<think>', '🤔 **Thinking:**\n', text)
    text = re.sub(r'</think>', '\n\n💡 **Response:**\n', text)
    
    return text

# ============================================================================
# TELEGRAM BOT FUNCTIONALITY
# ============================================================================

# Global user data storage
user_data = {}

def get_user_data(user_id):
    """Get or initialize user data for a given user ID"""
    if user_id not in user_data:
        user_data[user_id] = {
            'llama_history': [],
            'deepseek_history': [],
            'last_model': None,
            'auto_research': True,  # Default to enabled
            'conversation_mode': 'none'  # 'llama', 'deepseek', or 'none'
        }
    # Ensure all required fields exist for existing users
    if 'auto_research' not in user_data[user_id]:
        user_data[user_id]['auto_research'] = True
    if 'conversation_mode' not in user_data[user_id]:
        user_data[user_id]['conversation_mode'] = 'none'
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

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def start(update, context):
    """Handle /start command - welcome message with new arXiv capabilities"""
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
        "🧠 Smart Chat (Recommended):\n" +
        "/llama <question> - Chat with LLAMA 3.1\n" +
        "/deepseek <question> - Chat with Deepseek R1\n" +
        "💡 Just ask naturally! AI will automatically search, download, and analyze papers when needed.\n\n" +
        "📚 Enhanced Research Tools:\n" +
        "/search <query> - Search ArXiv papers with advanced filters\n" +
        "/download <paper_id> - Download papers for detailed analysis\n" +
        "/list - View all downloaded papers\n" +
        "/read <paper_id> - Read full paper content\n" +
        "/analyze <paper_id> - Get comprehensive paper analysis\n" +
        "/topics - List available research topics\n\n" +
        "🔧 Utility Commands:\n" +
        "/reset - Clear conversation history\n" +
        "/help - Show detailed help and features\n\n" +
        "✨ New Features: Advanced paper search with date/category filters, paper downloading, full-text reading, and AI-powered deep analysis!\n\n" +
        "👇 Choose your AI assistant below to get started!",
        reply_markup=reply_markup
    )
    
    # Show trending research topics as clickable buttons for new users
    send_onboarding_research_suggestions(update)

def help_command(update, context):
    """Handle /help command - detailed help with new features"""
    user_id = update.effective_user.id
    reply_markup = update_keyboard(user_id)
    
    send_telegram_message(update,
        "🤖 Inquisita Spark Research Assistant Help\n\n" +
        "🧠 Smart Chat (Recommended):\n" +
        "/llama <question> - Chat with LLAMA 3.1 (70B parameters)\n" +
        "/deepseek <question> - Chat with Deepseek R1 (advanced reasoning)\n" +
        "💡 Just ask naturally! AI automatically uses research tools when needed.\n\n" +
        "📚 Enhanced Research Tools:\n" +
        "/search <query> - Search ArXiv papers with advanced filters\n" +
        "   • Supports date filters (e.g., 'quantum computing since 2023')\n" +
        "   • Category filters (e.g., 'cs.AI', 'cs.LG')\n" +
        "/download <paper_id> - Download papers for detailed analysis\n" +
        "/list - View all downloaded papers in your library\n" +
        "/read <paper_id> - Read full paper content and metadata\n" +
        "/analyze <paper_id> - Get comprehensive AI-powered analysis\n" +
        "/topics - List available research topics\n\n" +
        "🔧 Utility Commands:\n" +
        "/reset - Clear conversation history\n" +
        "/help - Show this help message\n\n" +
        "✨ New Features in v2.0:\n" +
        "• Advanced paper search with date/category filters\n" +
        "• Paper downloading and local storage\n" +
        "• Full-text paper reading capabilities\n" +
        "• AI-powered deep paper analysis\n" +
        "• Seamless LLM integration with research tools\n\n" +
        "🔬 Auto-Research: Toggle below to enable/disable automatic tool usage during conversations.",
        reply_markup=reply_markup
    )

def search_command(update, context):
    """Handle /search command - search ArXiv papers with new API"""
    user_id = update.effective_user.id
    args = context.args
    if not args:
        send_telegram_message(update, "📚 **Search ArXiv Papers**\n\nUsage: `/search <query>`\n\nExample: `/search quantum computing`\n\n💡 Supports advanced filters like date ranges and categories!")
        return
    
    query = " ".join(args)
    try:
        # Show progress message
        progress_msg = send_telegram_message(update, f"🔍 Searching ArXiv for '{query}'...")
        
        paper_ids = search_papers(query, 10)  # Use new search_papers function
        
        if paper_ids:
            response = f"📚 **Search Results for '{query}':**\n\n"
            
            # Get detailed info for each paper
            for i, paper_id in enumerate(paper_ids[:10], 1):
                try:
                    paper_info = extract_info(paper_id)
                    if isinstance(paper_info, dict) and 'error' not in paper_info:
                        title = paper_info.get('title', 'Unknown Title')[:80] + ('...' if len(paper_info.get('title', '')) > 80 else '')
                        authors = ', '.join(paper_info.get('authors', [])[:2])
                        if len(paper_info.get('authors', [])) > 2:
                            authors += ' et al.'
                        
                        published = paper_info.get('published', 'Unknown')
                        pdf_url = paper_info.get('pdf_url', '')
                        
                        response += f"{i}. **{title}**\n"
                        response += f"   📄 ID: `{paper_id}`\n"
                        response += f"   👥 Authors: {authors} ({published})\n"
                        if pdf_url:
                            response += f"   [📄 PDF]({pdf_url})\n"
                        response += f"\n"
                    else:
                        response += f"{i}. Paper ID: `{paper_id}`\n\n"
                except Exception:
                    response += f"{i}. Paper ID: `{paper_id}`\n\n"
            
            response += f"💡 **Next steps:**\n"
            response += f"• Use `/download <paper_id>` to download papers\n"
            response += f"• Use `/analyze <paper_id>` for AI analysis"
        else:
            response = f"❌ No papers found for '{query}'. Try a different research area or broader terms."
        
        send_telegram_message(update, response)
        
    except Exception as e:
        print(f"Error in search_command: {str(e)}")
        send_telegram_message(update, f"❌ Error searching for papers on '{query}': {str(e)}")

def download_command(update, context):
    """Download a paper by arXiv ID for detailed analysis"""
    args = context.args
    if not args:
        send_telegram_message(update, "📥 **Download Paper**\n\nUsage: `/download <paper_id>`\n\nExample: `/download 2301.12345`\n\n💡 This downloads the paper for detailed reading and analysis.")
        return
    
    paper_id = args[0]
    try:
        # Show progress message
        progress_msg = send_telegram_message(update, f"📥 Downloading paper {paper_id}...")
        
        result = download_paper(paper_id)
        
        if "error" in result:
            send_telegram_message(update, f"❌ Error downloading paper: {result['error']}")
            return
        
        # Format success message
        msg = f"✅ **Paper Downloaded Successfully!**\n\n"
        msg += f"📄 **Title:** {result.get('title', 'N/A')}\n"
        msg += f"👥 **Authors:** {', '.join(result.get('authors', []))}\n"
        msg += f"📅 **Published:** {result.get('published', 'N/A')}\n\n"
        msg += f"📚 The paper is now available for reading and analysis.\n\n"
        msg += f"💡 **Next steps:**\n"
        msg += f"• Use `/read {paper_id}` to read the full content\n"
        msg += f"• Use `/analyze {paper_id}` for AI-powered analysis"
        
        send_telegram_message(update, msg)
        
    except Exception as e:
        print(f"Error in download_command: {str(e)}")
        send_telegram_message(update, f"❌ Error downloading paper: {str(e)}")

def list_command(update, context):
    """List all downloaded papers"""
    try:
        # Show progress message
        progress_msg = send_telegram_message(update, "📚 Retrieving your paper library...")
        
        result = list_papers()
        
        if "error" in result:
            send_telegram_message(update, f"❌ Error retrieving papers: {result['error']}")
            return
        
        papers = result.get('papers', [])
        
        if not papers:
            send_telegram_message(update, "📭 **No Papers Downloaded Yet**\n\nUse `/download <paper_id>` to download papers for your library.\n\n💡 Try searching first with `/search <topic>` to find interesting papers!")
            return
        
        # Format the papers list
        msg = f"📚 **Your Paper Library** ({len(papers)} papers)\n\n"
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Unknown Title')
            paper_id = paper.get('id', 'Unknown ID')
            authors = paper.get('authors', [])
            published = paper.get('published', 'Unknown Date')
            
            # Truncate title if too long
            if len(title) > 60:
                title = title[:57] + "..."
            
            # Truncate authors if too many
            author_str = ', '.join(authors[:2])
            if len(authors) > 2:
                author_str += f" et al. ({len(authors)} total)"
            
            msg += f"**{i}.** {title}\n"
            msg += f"   📄 ID: `{paper_id}`\n"
            msg += f"   👥 {author_str}\n"
            msg += f"   📅 {published}\n\n"
        
        msg += f"💡 **Commands:**\n"
        msg += f"• `/read <paper_id>` - Read full content\n"
        msg += f"• `/analyze <paper_id>` - Get AI analysis"
        
        send_telegram_message(update, msg)
        
    except Exception as e:
        print(f"Error in list_command: {str(e)}")
        send_telegram_message(update, f"❌ Error retrieving paper list: {str(e)}")

def read_command(update, context):
    """Read the full content of a downloaded paper"""
    args = context.args
    if not args:
        send_telegram_message(update, "📖 **Read Paper**\n\nUsage: `/read <paper_id>`\n\nExample: `/read 2301.12345`\n\n💡 This shows the full content of a downloaded paper.")
        return
    
    paper_id = args[0]
    try:
        # Show progress message
        progress_msg = send_telegram_message(update, f"📖 Reading paper {paper_id}...")
        
        result = read_paper(paper_id)
        
        if "error" in result:
            send_telegram_message(update, f"❌ Error reading paper: {result['error']}\n\n💡 Make sure the paper is downloaded first using `/download {paper_id}`")
            return
        
        # Format the paper content
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        
        # Start with metadata
        msg = f"📖 **Paper Content**\n\n"
        msg += f"📄 **Title:** {metadata.get('title', 'N/A')}\n"
        msg += f"👥 **Authors:** {', '.join(metadata.get('authors', []))}\n"
        msg += f"📅 **Published:** {metadata.get('published', 'N/A')}\n"
        msg += f"🔗 **URL:** {metadata.get('url', 'N/A')}\n\n"
        
        # Add content (truncate if too long for Telegram)
        if content:
            msg += f"📝 **Content:**\n{content}"
        else:
            msg += "📝 **Content:** No content available"
        
        # Check if message is too long for Telegram (4096 char limit)
        if len(msg) > 4000:
            # Split into chunks
            header = msg[:msg.find("📝 **Content:**")]
            content_start = msg.find("📝 **Content:**")
            
            # Send header first
            send_telegram_message(update, header)
            
            # Send content in chunks
            content_text = msg[content_start:]
            chunk_size = 4000
            for i in range(0, len(content_text), chunk_size):
                chunk = content_text[i:i + chunk_size]
                if i == 0:
                    send_telegram_message(update, chunk)
                else:
                    send_telegram_message(update, f"📝 **Content (continued):**\n{chunk}")
        else:
            send_telegram_message(update, msg)
        
        # Add follow-up suggestions
        follow_up = f"\n💡 **Next steps:**\n• Use `/analyze {paper_id}` for AI-powered analysis\n• Ask me questions about this paper!"
        send_telegram_message(update, follow_up)
        
    except Exception as e:
        print(f"Error in read_command: {str(e)}")
        send_telegram_message(update, f"❌ Error reading paper: {str(e)}")

def analyze_command(update, context):
    """Get comprehensive AI-powered analysis of a downloaded paper"""
    args = context.args
    if not args:
        send_telegram_message(update, "🔬 **Analyze Paper**\n\nUsage: `/analyze <paper_id>`\n\nExample: `/analyze 2301.12345`\n\n💡 This provides comprehensive AI-powered analysis of a downloaded paper.")
        return
    
    paper_id = args[0]
    try:
        # Show progress message
        progress_msg = send_telegram_message(update, f"🔬 Analyzing paper {paper_id} with AI...\n\n⏳ This may take a moment for comprehensive analysis.")
        
        result = get_deep_paper_analysis(paper_id)
        
        if "error" in result:
            send_telegram_message(update, f"❌ Error analyzing paper: {result['error']}\n\n💡 Make sure the paper is downloaded first using `/download {paper_id}`")
            return
        
        # Format the analysis
        analysis = result.get('analysis', '')
        metadata = result.get('metadata', {})
        
        # Start with paper info
        msg = f"🔬 **AI Paper Analysis**\n\n"
        msg += f"📄 **Paper:** {metadata.get('title', paper_id)}\n"
        msg += f"👥 **Authors:** {', '.join(metadata.get('authors', []))}\n\n"
        
        # Add analysis
        if analysis:
            msg += f"📊 **Analysis:**\n{analysis}"
        else:
            msg += "📊 **Analysis:** No analysis available"
        
        # Check if message is too long for Telegram
        if len(msg) > 4000:
            # Split into chunks
            header = msg[:msg.find("📊 **Analysis:**")]
            analysis_start = msg.find("📊 **Analysis:**")
            
            # Send header first
            send_telegram_message(update, header)
            
            # Send analysis in chunks
            analysis_text = msg[analysis_start:]
            chunk_size = 4000
            for i in range(0, len(analysis_text), chunk_size):
                chunk = analysis_text[i:i + chunk_size]
                if i == 0:
                    send_telegram_message(update, chunk)
                else:
                    send_telegram_message(update, f"📊 **Analysis (continued):**\n{chunk}")
        else:
            send_telegram_message(update, msg)
        
        # Add follow-up suggestions
        follow_up = f"\n💡 **Follow up:**\n• Ask me specific questions about this analysis\n• Use `/read {paper_id}` to see the full paper content"
        send_telegram_message(update, follow_up)
        
    except Exception as e:
        print(f"Error in analyze_command: {str(e)}")
        send_telegram_message(update, f"❌ Error analyzing paper: {str(e)}")

def topics_command(update, context):
    """Handle /topics command - list available research topics"""
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

def papers_command(update, context):
    """Handle /papers command - list papers in a topic"""
    args = context.args
    if not args:
        send_telegram_message(update, "📚 **View Papers by Topic**\n\nUsage: `/papers <topic>`\n\nExample: `/papers quantum computing`\n\n💡 Use `/topics` to see available topics.")
        return
    
    topic = " ".join(args)
    try:
        papers_data = get_topic_papers(topic)
        if not papers_data:
            send_telegram_message(update, f"📭 No papers found for topic '{topic}'.\n\n💡 Try `/search {topic}` to find and add papers to this topic.")
            return
        
        response = f"📚 **Papers on '{topic}'** ({len(papers_data)} found):\n\n"
        for i, paper in enumerate(papers_data, 1):
            title = paper.get('title', 'Unknown Title')
            if len(title) > 60:
                title = title[:57] + "..."
            
            response += f"{i}. **{title}**\n"
            response += f"   📄 ID: `{paper.get('id', 'Unknown ID')}`\n"
            response += f"   👥 Authors: {', '.join(paper.get('authors', [])[:2])}{'...' if len(paper.get('authors', [])) > 2 else ''}\n"
            response += f"   📅 Published: {paper.get('published', 'Unknown')[:10]}\n\n"
        
        response += "💡 **Commands:**\n"
        response += "• `/download <paper_id>` - Download for analysis\n"
        response += "• `/paper <paper_id>` - Get detailed info"
        send_telegram_message(update, response)
    except Exception as e:
        print(f"Error in papers_command: {str(e)}")
        send_telegram_message(update, f"❌ Error retrieving papers: {str(e)}")

def paper_command(update, context):
    """Handle /paper command - get detailed paper info"""
    args = context.args
    if not args:
        send_telegram_message(update, "📄 **Get Paper Info**\n\nUsage: `/paper <paper_id>`\n\nExample: `/paper 2301.12345`\n\n💡 This shows detailed information about a specific paper.")
        return
    
    paper_id = args[0]
    try:
        info = extract_info(paper_id)
        if "error" in info:
            send_telegram_message(update, f"❌ Error: {info['error']}")
            return
        
        msg = f"📄 **Paper Information**\n\n"
        msg += f"📄 **ID:** `{paper_id}`\n"
        msg += f"📝 **Title:** {info.get('title', 'N/A')}\n"
        msg += f"👥 **Authors:** {', '.join(info.get('authors', []))}\n"
        msg += f"📅 **Published:** {info.get('published', 'N/A')}\n"
        if info.get('pdf_url'):
            msg += f"🔗 **PDF:** [Download]({info['pdf_url']})\n"
        msg += f"\n📋 **Abstract:**\n{info.get('summary', 'N/A')}\n\n"
        msg += f"💡 **Next steps:**\n"
        msg += f"• Use `/download {paper_id}` to download for analysis\n"
        msg += f"• Use `/analyze {paper_id}` for AI-powered insights"
        
        send_telegram_message(update, msg)
    except Exception as e:
        print(f"Error in paper_command: {str(e)}")
        send_telegram_message(update, f"❌ Error retrieving paper info: {str(e)}")

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

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

def llama_command(update, context):
    """Handle /llama command - chat with LLAMA 3.1"""
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
    reply = get_llama_reply(udata['llama_history'], enable_tools=auto_research_enabled, update=update)
    
    # Only add assistant message to history if it's not an error
    if not reply.startswith("⚠️"):
        udata['llama_history'].append({"role": "assistant", "content": reply})
    
    # Send the main response
    send_telegram_message(update, reply)
    
    # Send interactive hints as separate messages
    if not reply.startswith("⚠️"):  # Only send hints for successful responses
        send_interactive_hints(update, reply)

def deepseek_command(update, context):
    """Handle /deepseek command - chat with Deepseek R1"""
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
    reply = get_deepseek_reply(udata['deepseek_history'], enable_tools=auto_research_enabled, update=update)
    
    # Only add assistant message to history if it's not an error
    if not reply.startswith("⚠️"):
        udata['deepseek_history'].append({"role": "assistant", "content": reply})
    
    # Send the main response
    send_telegram_message(update, reply)
    
    # Send interactive hints as separate messages
    if not reply.startswith("⚠️"):  # Only send hints for successful responses
        send_interactive_hints(update, reply)

def reset_command(update, context):
    """Handle /reset command - clear conversation history"""
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

def message_handler(update, context):
    """Handle text messages and route to appropriate AI assistant"""
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
    
    # Handle keyboard button presses for AI assistant selection
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

# Placeholder functions for onboarding and interactive hints
# These functions are referenced in the command handlers but need to be implemented
def send_onboarding_research_suggestions(update):
    """Send trending research topics as clickable buttons for new users"""
    # This function would generate and send research topic suggestions
    # For now, we'll implement a simple version
    try:
        # Generate some default research topics
        topics = [
            "🤖 AI Research",
            "⚛️ Quantum Computing", 
            "🧬 Biotech Advances",
            "🌍 Climate Science"
        ]
        
        keyboard = []
        for i in range(0, len(topics), 2):
            row = [InlineKeyboardButton(topics[i], callback_data=f"topic_{i}")]
            if i + 1 < len(topics):
                row.append(InlineKeyboardButton(topics[i + 1], callback_data=f"topic_{i+1}"))
            keyboard.append(row)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        send_telegram_message(
            update,
            "🔬 **Trending Research Topics:**\n\nClick any topic below to start exploring!",
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"Error in send_onboarding_research_suggestions: {str(e)}")
        # Fail silently to not disrupt user experience
        pass

def send_interactive_hints(update, reply, tools_used=None):
    """Send interactive hints as buttons after AI responses"""
    # This function would generate contextual follow-up suggestions
    # For now, we'll implement a simple version with direct tool access
    try:
        user_id = update.effective_user.id
        
        # Create direct tool action buttons
        keyboard = [
            [
                InlineKeyboardButton("🔍 Search Papers", callback_data=f"direct_search_{user_id}"),
                InlineKeyboardButton("📚 Browse Topics", callback_data=f"direct_topics_{user_id}")
            ],
            [
                InlineKeyboardButton("📄 Paper Info", callback_data=f"direct_paper_{user_id}"),
                InlineKeyboardButton("🗺️ Research Guide", callback_data=f"direct_guide_{user_id}")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        send_telegram_message(
            update,
            "💡 **Quick Actions:**\n\nChoose what you'd like to explore next!",
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"Error in send_interactive_hints: {str(e)}")
        # Fail silently to not disrupt user experience
        pass

def handle_hint_callback(update, context):
    """Handle callback queries from interactive hint buttons"""
    query = update.callback_query
    user_id = query.from_user.id
    callback_data = query.data
    
    try:
        # Answer the callback query to remove loading state
        query.answer()
        
        # Handle different callback types
        if callback_data.startswith("direct_"):
            action = callback_data.split("_")[1]
            
            if action == "search":
                send_telegram_message(query, "🔍 Use `/search <topic>` to find papers!\n\nExample: `/search quantum computing`")
            elif action == "topics":
                # Trigger topics command
                context.args = []
                topics_command(query, context)
            elif action == "paper":
                send_telegram_message(query, "📄 Use `/paper <paper_id>` to get detailed info!\n\nExample: `/paper 2301.12345`")
            elif action == "guide":
                send_telegram_message(query, "🗺️ Use `/help` to see all available commands and features!")
        
        elif callback_data.startswith("topic_"):
            # Handle topic selection from onboarding
            topic_index = int(callback_data.split("_")[1])
            topics = ["AI Research", "Quantum Computing", "Biotech Advances", "Climate Science"]
            
            if topic_index < len(topics):
                topic = topics[topic_index]
                send_telegram_message(query, f"🔍 Searching for papers on {topic}...")
                # Trigger search command
                context.args = topic.lower().split()
                search_command(query, context)
    
    except Exception as e:
        print(f"Error in handle_hint_callback: {str(e)}")
        query.answer("Sorry, there was an error processing your request.")

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
