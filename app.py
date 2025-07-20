import os
import json
import requests
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

RESEARCH_SERVER_URL = "https://mcp-arxiv-server.onrender.com"

# ============================================================================
# RESEARCH FUNCTIONALITY (via remote MCP server)
# ============================================================================

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Call the remote research server to search for papers on a topic.
    Returns: List of paper IDs
    """
    resp = requests.post(f"{RESEARCH_SERVER_URL}/search_papers", json={"topic": topic, "max_results": max_results})
    resp.raise_for_status()
    return resp.json().get("paper_ids", [])

def extract_info(paper_id: str):
    """
    Call the remote research server to get info about a specific paper.
    Returns: JSON string with paper info
    """
    resp = requests.get(f"{RESEARCH_SERVER_URL}/extract_info", params={"paper_id": paper_id})
    resp.raise_for_status()
    return resp.json()

def get_available_folders():
    """
    Call the remote research server to list all available topic folders.
    Returns: List of topic folder names
    """
    resp = requests.get(f"{RESEARCH_SERVER_URL}/get_available_folders")
    resp.raise_for_status()
    return resp.json().get("topics", [])

def get_topic_papers(topic: str):
    """
    Call the remote research server to get all papers for a topic.
    Returns: JSON string with all papers info
    """
    resp = requests.get(f"{RESEARCH_SERVER_URL}/get_topic_papers", params={"topic": topic})
    resp.raise_for_status()
    return resp.json()

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
        papers_info = get_topic_papers(topic)
        if not papers_info:
            send_telegram_message(update, f"No papers found for topic '{topic}'.")
            return
        papers_data = papers_info
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
        msg += f"Abstract: {info.get('abstract', 'N/A')}\n"
        msg += f"URL: {info.get('url', 'N/A')}\n"
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
        send_telegram_message(update, "Please enter a research topic to search for.\nExample: machine learning, quantum computing, neural networks")
        return
    elif text == "View Papers by Topic":
        return topics_command(update, context)
    elif text == "List Topics":
        return topics_command(update, context)
    
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
