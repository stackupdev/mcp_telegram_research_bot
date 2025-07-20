import os
import json
import asyncio
from typing import List
from flask import Flask, render_template, request, jsonify
from groq import Groq
from telegram import Update, Bot, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
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

# ============================================================================
# LLM FUNCTIONALITY (Groq API)
# ============================================================================

def get_llama_reply(messages: list) -> str:
    try:
        client = Groq()
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        error_str = str(e)
        print(f"Error in get_llama_reply: {error_str}")
        
        # Handle token limit errors
        if "413" in error_str and "Request too large" in error_str:
            return "⚠️ Your conversation history is too long for the model's token limit. Please use /reset to start a new conversation, or ask a shorter question."
        
        return f"⚠️ Error from Groq API: {error_str}"

def get_deepseek_reply(messages: list) -> str:
    try:
        client = Groq()
        completion_ds = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=messages
        )
        return completion_ds.choices[0].message.content
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
        user_data[user_id] = {}
    return user_data[user_id]

def truncate_conversation(messages, max_tokens=MAX_TOKENS):
    """
    Automatically truncate conversation history to stay within token limits.
    Uses a simple heuristic of ~4 chars per token for estimation.
    """
    
    if not messages:
        return messages
    
    # Estimate tokens (rough heuristic: ~4 characters per token)
    total_chars = sum(len(msg.get('content', '')) for msg in messages)
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
        msg_chars = len(msg.get('content', ''))
        if (chars_count + msg_chars) // 4 > max_tokens:
            break
        truncated.insert(-1 if truncated and truncated[0].get('role') == 'system' else 0, msg)
        chars_count += msg_chars
    
    return truncated

def send_telegram_message(update, text, reply_markup=None):
    """Split long messages into smaller chunks to avoid Telegram's 4096 character limit"""
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
    
    # Create custom keyboard with LLM chat and MCP tools toggle
    keyboard = [
        [KeyboardButton("Chat with LLAMA"), KeyboardButton("Chat with Deepseek")],
        [KeyboardButton("MCP Tools")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    send_telegram_message(
        update,
        f"Welcome to Inquisita Spark Research Bot, {user_name}!\n\n" +
        "🔬 Academic research and AI chat features:\n" +
        "• Chat with LLAMA or Deepseek\n" +
        "• MCP Tools: Search papers, view by topic, list topics\n\n" +
        "Select an option below or use commands:\n" +
        "/search <topic> - Search academic papers\n" +
        "/papers <topic> - View papers for a topic\n" +
        "/topics - List available research topics\n" +
        "/llama <question> - Chat with LLAMA\n" +
        "/deepseek <question> - Chat with Deepseek\n",
        reply_markup=reply_markup
    )

def help_command(update, context):
    send_telegram_message(update,
        "🤖 Inquisita Spark Research Bot Commands:\n\n" +
        "📚 Research Commands:\n" +
        "/search <topic> - Search ArXiv papers\n" +
        "/papers <topic> - View papers for topic\n" +
        "/paper <id> - Get specific paper details\n" +
        "/topics - List available topics\n\n" +
        "🧠 AI Chat Commands:\n" +
        "/llama <question> - Chat with LLAMA AI\n" +
        "/deepseek <question> - Chat with Deepseek AI\n" +
        "/reset - Reset conversation history\n"
    )

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
            send_telegram_message(update, "No topics available.")
            return
        msg = "Available topics:\n" + "\n".join(topics)
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
        send_telegram_message(update, "Please provide a question after /llama")
        return
        
    q = ' '.join(context.args)
    print(f"LLAMA query from user {user_id}: {q}")
    
    # Add user message to history
    udata['llama_history'].append({"role": "user", "content": q})
    
    # Truncate conversation if needed before sending to API
    udata['llama_history'] = truncate_conversation(udata['llama_history'])
    
    # Get reply from LLAMA
    reply = get_llama_reply(udata['llama_history'])
    
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
        send_telegram_message(update, "Please provide a question after /deepseek")
        return
        
    q = ' '.join(context.args)
    print(f"Deepseek query from user {user_id}: {q}")
    
    # Add user message to history
    udata['deepseek_history'].append({"role": "user", "content": q})
    
    # Truncate conversation if needed before sending to API
    udata['deepseek_history'] = truncate_conversation(udata['deepseek_history'])
    
    # Get reply from Deepseek
    reply = get_deepseek_reply(udata['deepseek_history'])
    
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

def message_handler(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    text = update.message.text
    
    # Toggle MCP Tools keyboard
    if text == "MCP Tools":
        mcp_keyboard = [
            [KeyboardButton("Search Papers")],
            [KeyboardButton("View Papers by Topic")],
            [KeyboardButton("List Topics")],
            [KeyboardButton("Back to Main Menu")]
        ]
        reply_markup = ReplyKeyboardMarkup(mcp_keyboard, resize_keyboard=True)
        send_telegram_message(update, "MCP Tools: Select a research action.", reply_markup=reply_markup)
        return
    elif text == "Back to Main Menu":
        # Show main keyboard again
        main_keyboard = [
            [KeyboardButton("Chat with LLAMA"), KeyboardButton("Chat with Deepseek")],
            [KeyboardButton("MCP Tools")]
        ]
        reply_markup = ReplyKeyboardMarkup(main_keyboard, resize_keyboard=True)
        send_telegram_message(update, "Main menu:", reply_markup=reply_markup)
        return
    
    # Handle keyboard button presses
    if text == "Chat with LLAMA":
        context.args = ["Hi,", "I'd", "like", "to", "chat"]
        return llama_command(update, context)
    elif text == "Chat with Deepseek":
        context.args = ["Hi,", "I'd", "like", "to", "chat"]
        return deepseek_command(update, context)
    elif text == "Search Papers":
        # Set user state to expect search input
        udata['awaiting_search'] = True
        send_telegram_message(update, "Please enter a research topic to search for.\nExample: machine learning, quantum computing, neural networks")
        return
    elif text == "View Papers by Topic":
        # Set user state to expect topic input for viewing papers
        udata['awaiting_topic_selection'] = True
        topics = get_available_folders()
        if not topics:
            send_telegram_message(update, "No topics available. Search for papers first to create topics.")
            return
        topic_list = "\n".join([f"• {topic}" for topic in topics])
        send_telegram_message(update, f"Available topics:\n{topic_list}\n\nPlease enter the topic name to view papers:")
        return
    elif text == "List Topics":
        return topics_command(update, context)
    
    # Check if user is awaiting search input
    if udata.get('awaiting_search', False):
        udata['awaiting_search'] = False  # Clear the state
        context.args = [text]  # Set the search topic
        return search_command(update, context)
    
    # Check if user is awaiting topic selection for viewing papers
    if udata.get('awaiting_topic_selection', False):
        udata['awaiting_topic_selection'] = False  # Clear the state
        context.args = [text]  # Set the topic name
        return papers_command(update, context)
    
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
            messages = [{"role": "user", "content": q}]
            reply = get_llama_reply(messages)
            return render_template("llama_reply.html", r=reply)
    return render_template("llama.html")

@app.route('/deepseek', methods=['GET', 'POST'])
def deepseek():
    if request.method == 'POST':
        q = request.form.get("q")
        if q:
            messages = [{"role": "user", "content": q}]
    reply = get_deepseek_reply(messages)
    return render_template("deepseek_reply.html", r=reply)

if __name__ == "__main__":
    app.run(debug=True)
