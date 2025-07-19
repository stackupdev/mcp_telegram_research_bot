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

# ========== RESEARCH LOGIC SETUP ==========
from research_logic import research_processor, generate_research_prompt, create_research_summary

# ============================================================================
# RESEARCH FUNCTIONALITY (Direct Integration)
# ============================================================================
# Research functionality is now handled by research_logic.py module

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
        [KeyboardButton("🔍 Start Research"), KeyboardButton("Research Status")],
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
    help_text = (
        "🤖 **Available Commands:**\n\n"
        "**Chatbots:**\n"
        "/llama - Chat with LLAMA model\n"
        "/deepseek - Chat with Deepseek model\n\n"
        "**Deep Research:**\n"
        "/research <question> - Generate structured research prompt\n"
        "/get_prompt - Get full research prompt for AI assistants\n"
        "/research_status - View current research status\n"
        "/clear_research - Clear current research data\n\n"
        "**General:**\n"
        "/help - Show this help message\n"
        "/reset - Reset conversation history\n\n"
        "**How it works:**\n"
        "1. Use /research with your question\n"
        "2. Get a structured research prompt\n"
        "3. Use /get_prompt to copy the full prompt\n"
        "4. Paste it into Claude, ChatGPT, or similar AI tools\n"
    )
    send_telegram_message(update, help_text)

def telegram_research_command(update, context):
    """Handle /research command - generate deep research prompt"""
    if not context.args:
        send_telegram_message(update, "Please provide a research question.\nUsage: /research What are the latest developments in AI?")
        return
    
    research_question = ' '.join(context.args)
    
    try:
        # Generate research prompt
        prompt = generate_research_prompt(research_question)
        summary = create_research_summary(research_question)
        
        # Store research question
        research_processor.update_research_data("question", research_question)
        research_processor.add_note(f"Research initiated: {research_question}")
        
        # Send summary to user
        send_telegram_message(update, summary)
        
        # Send the research prompt (truncated for Telegram)
        prompt_preview = prompt[:1000] + "...\n\n📋 **Full research prompt generated!**\nUse /get_prompt to get the complete research prompt for use with AI assistants."
        send_telegram_message(update, f"**Research Prompt Preview:**\n\n{prompt_preview}")
        
    except Exception as e:
        print(f"Error in research_command: {str(e)}")
        send_telegram_message(update, f"❌ Error generating research prompt: {str(e)}")

def telegram_get_prompt_command(update, context):
    """Handle /get_prompt command - get full research prompt"""
    try:
        research_data = research_processor.get_research_data()
        current_question = research_data.get("question", "")
        
        if not current_question:
            send_telegram_message(update, "No active research question. Use /research <question> first.")
            return
        
        # Generate full prompt
        full_prompt = generate_research_prompt(current_question)
        
        # Split prompt into chunks for Telegram (max 4096 characters per message)
        chunk_size = 4000
        chunks = [full_prompt[i:i+chunk_size] for i in range(0, len(full_prompt), chunk_size)]
        
        send_telegram_message(update, f"📋 **Full Research Prompt for:** {current_question}\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            send_telegram_message(update, f"**Part {i}/{len(chunks)}:**\n\n{chunk}")
        
        send_telegram_message(update, "\n✅ **Complete!** Copy this prompt and use it with Claude, ChatGPT, or other AI assistants with web search capabilities.")
        
    except Exception as e:
        print(f"Error in get_prompt_command: {str(e)}")
        send_telegram_message(update, f"❌ Error retrieving research prompt: {str(e)}")

def telegram_research_status_command(update, context):
    """Handle /research_status command - show current research status"""
    try:
        research_data = research_processor.get_research_data()
        notes = research_processor.get_research_notes()
        
        current_question = research_data.get("question", "None")
        
        status_message = f"📊 **Research Status**\n\n"
        status_message += f"**Current Question:** {current_question}\n\n"
        
        if notes:
            recent_notes = notes.split('\n')[-5:]  # Last 5 notes
            status_message += f"**Recent Activity:**\n"
            for note in recent_notes:
                status_message += f"• {note}\n"
        else:
            status_message += "**Activity:** No research activity yet\n"
        
        status_message += f"\n**Available Commands:**\n"
        status_message += f"• /research <question> - Start new research\n"
        status_message += f"• /get_prompt - Get full research prompt\n"
        status_message += f"• /clear_research - Clear current research\n"
        
        send_telegram_message(update, status_message)
        
    except Exception as e:
        print(f"Error in research_status_command: {str(e)}")
        send_telegram_message(update, f"❌ Error retrieving research status: {str(e)}")

def telegram_clear_research_command(update, context):
    """Handle /clear_research command - clear current research data"""
    try:
        research_processor.clear_research()
        send_telegram_message(update, "✅ Research data cleared. You can start a new research with /research <question>")
        
    except Exception as e:
        print(f"Error in clear_research_command: {str(e)}")
        send_telegram_message(update, f"❌ Error clearing research data: {str(e)}")

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
    elif text == "Start Research":
        send_telegram_message(update, "Please enter a research question to investigate.\nExample: /research What are the latest developments in quantum computing?\n\nOr just type your question after clicking this button.")
        return
    elif text == "Research Status":
        return telegram_research_status_command(update, context)
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
    telegram_dispatcher.add_handler(CommandHandler("research", telegram_research_command))
    telegram_dispatcher.add_handler(CommandHandler("get_prompt", telegram_get_prompt_command))
    telegram_dispatcher.add_handler(CommandHandler("research_status", telegram_research_status_command))
    telegram_dispatcher.add_handler(CommandHandler("clear_research", telegram_clear_research_command))
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
    # Get current research status if any
    current_research = None
    if research_processor.research_data.get('question'):
        current_research = {
            'question': research_processor.research_data['question'],
            'status': 'active'
        }
    return render_template('research.html', research_status=current_research)

@app.route('/api/research', methods=['POST'])
def api_research():
    question = request.form.get('question')
    if not question:
        return render_template('research.html', error="Please provide a research question.")
    
    try:
        # Generate research prompt
        prompt = generate_research_prompt(question)
        summary = create_research_summary(question)
        
        # Update research processor
        research_processor.update_research_data("question", question)
        
        # Show preview (first 500 characters)
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        
        current_research = {
            'question': question,
            'status': 'active'
        }
        
        return render_template('research.html', 
                             research_status=current_research,
                             prompt_preview=prompt_preview,
                             current_question=question)
    except Exception as e:
        return render_template('research.html', error=f"Error generating research prompt: {str(e)}")

@app.route('/api/get_full_prompt', methods=['GET'])
def api_get_full_prompt():
    if not research_processor.research_data.get('question'):
        return render_template('research.html', error="No active research session. Please start a research first.")
    
    try:
        question = research_processor.research_data['question']
        prompt = generate_research_prompt(question)
        
        # Return as plain text for easy copying
        return f"<pre>{prompt}</pre>", 200, {'Content-Type': 'text/html'}
    except Exception as e:
        return render_template('research.html', error=f"Error retrieving prompt: {str(e)}")

@app.route('/api/clear_research', methods=['POST'])
def api_clear_research():
    try:
        # Clear research data
        research_processor.research_data = {
            "question": "",
            "elaboration": "",
            "subquestions": [],
            "search_results": {},
            "extracted_content": {},
            "final_report": "",
        }
        research_processor.notes = []
        
        return render_template('research.html', message="Research data cleared successfully.")
    except Exception as e:
        return render_template('research.html', error=f"Error clearing research: {str(e)}")

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
