import os
import json
import arxiv
from typing import List
from flask import Flask, render_template, request, jsonify
from groq import Groq
from telegram import Update, Bot, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters

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

# Ensure papers directory exists
os.makedirs(PAPER_DIR, exist_ok=True)

# ========== MCP CLIENT SETUP ==========
try:
    from mcp import MCPClient
    MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8001")
    mcp_client = MCPClient(MCP_SERVER_URL)
except ImportError:
    mcp_client = None
    print("[Warning] MCPClient not available. Install 'mcp' package for MCP integration.")

# ============================================================================
# RESEARCH FUNCTIONALITY (from inquisita_spark)
# ============================================================================

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_id = paper.entry_id.split('/')[-1]
        paper_ids.append(paper_id)
        
        # Store paper information
        papers_info[paper_id] = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "published": paper.published.isoformat() if paper.published else None,
            "pdf_url": paper.pdf_url,
            "entry_id": paper.entry_id,
            "categories": paper.categories
        }
    
    # Save updated papers info
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    return paper_ids

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
    
    # Search through all topic directories
    for topic_dir in os.listdir(PAPER_DIR):
        topic_path = os.path.join(PAPER_DIR, topic_dir)
        if os.path.isdir(topic_path):
            papers_file = os.path.join(topic_path, "papers_info.json")
            if os.path.exists(papers_file):
                try:
                    with open(papers_file, "r") as f:
                        papers_info = json.load(f)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
    
    return f"Paper with ID '{paper_id}' not found in any topic directory."

def get_available_folders() -> List[str]:
    """
    List all available topic folders in the papers directory.
    
    Returns:
        List of available topic folder names
    """
    
    folders = []
    if os.path.exists(PAPER_DIR):
        for item in os.listdir(PAPER_DIR):
            item_path = os.path.join(PAPER_DIR, item)
            if os.path.isdir(item_path):
                folders.append(item)
    
    return folders

def get_topic_papers(topic: str) -> str:
    """
    Get detailed information about papers on a specific topic.
    
    Args:
        topic: The research topic to retrieve papers for
        
    Returns:
        JSON string with all papers information for the topic
    """
    
    topic_folder = topic.lower().replace(" ", "_")
    topic_path = os.path.join(PAPER_DIR, topic_folder)
    papers_file = os.path.join(topic_path, "papers_info.json")
    
    if not os.path.exists(papers_file):
        return f"No papers found for topic '{topic}'. Use search_papers to find papers first."
    
    try:
        with open(papers_file, "r") as f:
            papers_info = json.load(f)
            return json.dumps(papers_info, indent=2)
    except (FileNotFoundError, json.JSONDecodeError):
        return f"Error reading papers information for topic '{topic}'."

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
    
    # Create custom keyboard with main options
    keyboard = [
        [KeyboardButton("Chat with LLAMA"), KeyboardButton("Chat with Deepseek")],
        [KeyboardButton("Search Papers"), KeyboardButton("View Topics")],
        [KeyboardButton("Reset Conversation")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    send_telegram_message(
        update,
        f"Welcome to Inquisita Spark Research Bot, {user_name}!\n\n" +
        "🔬 I can help you with:\n" +
        "• Academic paper research via ArXiv\n" +
        "• AI conversations with LLAMA & Deepseek\n" +
        "• Organizing research by topics\n\n" +
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

def telegram_search_command(update, context):
    topic = " ".join(context.args)
    if not mcp_client:
        update.message.reply_text("MCP client not available.")
        return
    try:
        result = mcp_client.resource("papers://search", topic=topic, max_results=5)
        update.message.reply_text(f"Search results for '{topic}':\n{result}")
    except Exception as e:
        update.message.reply_text(f"Error: {str(e)}")

def telegram_papers_command(update, context):
    topic = " ".join(context.args)
    if not mcp_client:
        update.message.reply_text("MCP client not available.")
        return
    try:
        result = mcp_client.resource("papers://topic", topic=topic)
        update.message.reply_text(f"Papers for '{topic}':\n{result}")
    except Exception as e:
        update.message.reply_text(f"Error: {str(e)}")

def telegram_paper_command(update, context):
    paper_id = " ".join(context.args)
    if not mcp_client:
        update.message.reply_text("MCP client not available.")
        return
    try:
        result = mcp_client.resource("papers://extract", paper_id=paper_id)
        update.message.reply_text(f"Paper info for '{paper_id}':\n{result}")
    except Exception as e:
        update.message.reply_text(f"Error: {str(e)}")

def telegram_topics_command(update, context):
    if not mcp_client:
        update.message.reply_text("MCP client not available.")
        return
    try:
        result = mcp_client.resource("papers://folders")
        update.message.reply_text(f"Available topics:\n{result}")
    except Exception as e:
        update.message.reply_text(f"Error: {str(e)}")

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
    
    # Handle keyboard button presses
    if text == "Chat with LLAMA":
        context.args = ["Hi,", "I'd", "like", "to", "chat"]
        return llama_command(update, context)
    elif text == "Chat with Deepseek":
        context.args = ["Hi,", "I'd", "like", "to", "chat"]
        return deepseek_command(update, context)
    elif text == "Search Papers":
        send_telegram_message(update, "Please enter a research topic to search for.\nExample: machine learning, quantum computing, neural networks")
        return
    elif text == "View Topics":
        return telegram_topics_command(update, context)
    elif text == "Reset Conversation":
        return reset_command(update, context)
    
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
    telegram_dispatcher.add_handler(CommandHandler("search", telegram_search_command))
    telegram_dispatcher.add_handler(CommandHandler("papers", telegram_papers_command))
    telegram_dispatcher.add_handler(CommandHandler("paper", telegram_paper_command))
    telegram_dispatcher.add_handler(CommandHandler("topics", telegram_topics_command))
    telegram_dispatcher.add_handler(CommandHandler("llama", llama_command))
    telegram_dispatcher.add_handler(CommandHandler("deepseek", deepseek_command))
    telegram_dispatcher.add_handler(CommandHandler("reset", reset_command))
    # Add handler for regular text messages (must be added last)
    telegram_dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, message_handler))

# ============================================================================
# FLASK WEB ROUTES
# ============================================================================

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    try:
        data = request.get_json(force=True)
        print("Received Telegram update:", data)
        
        # Check if we have a valid token
        if not TELEGRAM_BOT_TOKEN:
            print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set!")
            return jsonify(success=False, error="Bot token not configured"), 500
            
        update = Update.de_json(data, TELEGRAM_BOT)
        if update:
            print(f"Processing update ID: {update.update_id}")
            telegram_dispatcher.process_update(update)
        else:
            print("WARNING: Received invalid update format from Telegram")
            
        return jsonify(success=True)
    except Exception as e:
        print(f"ERROR in telegram_webhook: {str(e)}")
        return jsonify(success=False, error=str(e)), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
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

@app.route('/llama')
def llama():
    return render_template("llama.html")

@app.route('/llama_reply', methods=["POST"])
def llama_reply():
    q = request.form.get("q")
    messages = [{"role": "user", "content": q}]
    reply = get_llama_reply(messages)
    return render_template("llama_reply.html", r=reply)

@app.route('/deepseek')
def deepseek():
    return render_template("deepseek.html")

@app.route('/deepseek_reply', methods=["POST"])
def deepseek_reply():
    q = request.form.get("q")
    messages = [{"role": "user", "content": q}]
    reply = get_deepseek_reply(messages)
    return render_template("deepseek_reply.html", r=reply)

if __name__ == "__main__":
    app.run(debug=True)
