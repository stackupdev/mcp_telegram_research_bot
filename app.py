import os
from flask import Flask, render_template, request, jsonify
import joblib
from groq import Groq
from telegram import Update, Bot, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CommandHandler

# NOTE: Do NOT set your GROQ_API_KEY in code.
# Instead, set the GROQ_API_KEY as an environment variable in your Render.com dashboard:
# - Go to your service > Environment > Add Environment Variable
# - Key: GROQ_API_KEY, Value: <your_actual_api_key>
# The Groq client will automatically use this environment variable.

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_BOT = Bot(token=TELEGRAM_BOT_TOKEN)
# We'll use a global dispatcher for all updates
telegram_dispatcher = Dispatcher(TELEGRAM_BOT, None, workers=0, use_context=True)
# In-memory user_data for session context per user
user_data = {}

# Maximum tokens to allow in conversation history before truncating
MAX_TOKENS = 4000  # Conservative limit below Groq's 6000 token limit

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
            # Conversation history too long
            return "⚠️ Your conversation history is too long for the model's token limit. Please use /reset to start a new conversation, or ask a shorter question."

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
            # Conversation history too long
            return "⚠️ Your conversation history is too long for the model's token limit. Please use /reset to start a new conversation, or ask a shorter question."
        
        # Handle other API errors
        return f"⚠️ Error from Groq API: {error_str}"

def predict_dbs(usdsgd: float) -> str:
    try:
        # Get absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'dbs.jl')
        
        # Log the path for debugging
        print(f"Looking for model at: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            return "Prediction model not found. Please check server logs."
            
        # Try to load the model
        print("Loading model...")
        dbs_model = joblib.load(model_path)
        
        # Make prediction
        print(f"Making prediction with USD/SGD rate: {usdsgd}")
        pred = dbs_model.predict([[usdsgd]])[0]
        # Convert numpy array value to Python float before formatting
        pred_float = float(pred)
        return f"Predicted DBS share price: {pred_float:.2f} SGD"
        
    except Exception as e:
        print(f"ERROR in predict_dbs: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"Error making prediction: {str(e)}"

# Telegram command handlers

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
        
    # Simple estimation: ~4 chars per token
    total_chars = sum(len(msg["content"]) for msg in messages)
    estimated_tokens = total_chars // 4
    
    # If we're under the limit, no need to truncate
    if estimated_tokens <= max_tokens:
        return messages
        
    print(f"Truncating conversation: {estimated_tokens} tokens (estimated) exceeds {max_tokens} limit")
    
    # Keep truncating from the beginning until we're under the limit
    # Always keep at least the most recent exchange (2 messages)
    while estimated_tokens > max_tokens and len(messages) > 2:
        # If first message is system, remove the second message instead
        if messages and messages[0]["role"] == "system":
            if len(messages) <= 2:  # Only system + 1 message left
                break
            removed = messages.pop(1)
        else:
            removed = messages.pop(0)
            
        estimated_tokens -= len(removed["content"]) // 4
        print(f"Removed message: {removed['role']} ({len(removed['content']) // 4} tokens)")
    
    # Add a system message indicating truncation if we removed messages
    if estimated_tokens > max_tokens:
        truncation_notice = {"role": "system", "content": "[Some earlier messages were removed to stay within token limits]"}
        if messages and messages[0]["role"] == "system":
            # Insert after existing system message
            messages.insert(1, truncation_notice)
        else:
            # Insert at beginning
            messages.insert(0, truncation_notice)
            
    return messages

def send_telegram_message(update, text, reply_markup=None):
    """Split long messages into smaller chunks to avoid Telegram's 4096 character limit"""
    # Maximum message length for Telegram
    max_length = 4000  # Slightly less than 4096 to be safe
    
    # If the message is short enough, send it as is
    if len(text) <= max_length:
        update.message.reply_text(text, reply_markup=reply_markup)
        return
        
    # Otherwise, split it into chunks
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    # Send the first chunk with reply_markup
    update.message.reply_text(chunks[0], reply_markup=reply_markup)
    
    # Send the rest with a prefix, but no reply_markup
    prefix = "(continued) "
    for chunk in chunks[1:]:
        if len(chunk) + len(prefix) <= max_length:
            update.message.reply_text(prefix + chunk)
        else:
            update.message.reply_text(chunk)

def start(update, context):
    # Create custom keyboard with main options
    keyboard = [
        [KeyboardButton("Chat with LLAMA"), KeyboardButton("Chat with Deepseek")],
        [KeyboardButton("Predict DBS Price"), KeyboardButton("Reset Conversation")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    send_telegram_message(
        update,
        "Welcome to GroqSeeker_Bot!\n\n" +
        "Select an option below or type your question directly.\n\n" +
        "You can also use commands:\n" +
        "/llama <question> - Chat with LLAMA\n" +
        "/deepseek <question> - Chat with Deepseek\n" +
        "/predict <usdsgd> - Predict DBS price",
        reply_markup=reply_markup
    )

def help_command(update, context):
    send_telegram_message(update,
        "Commands:\n" +
        "/llama <question> - Chat with LLAMA AI\n" +
        "/deepseek <question> - Chat with Deepseek AI\n" +
        "/predict <usdsgd> - Predict DBS share price\n"
    )

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

def predict_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    
    # Debug print
    print(f"Predict command received from user {user_id}")
    print(f"Args: {context.args}")
    
    if not context.args:
        last_rate = udata.get('last_usdsgd')
        if last_rate:
            send_telegram_message(update, f"Your last USD/SGD rate was: {last_rate}")
        else:
            send_telegram_message(update, "Please provide the USD/SGD rate after /predict.")
        return
    try:
        usdsgd = float(context.args[0])
        print(f"Valid USD/SGD rate provided: {usdsgd}")
        udata['last_usdsgd'] = usdsgd
        reply = predict_dbs(usdsgd)
    except ValueError as e:
        print(f"Invalid input: {context.args[0]} - {str(e)}")
        reply = "Invalid input. Please provide a valid number for USD/SGD."
    except Exception as e:
        print(f"Error in predict_command: {str(e)}")
        reply = f"Error processing prediction: {str(e)}"
    update.message.reply_text(reply)

def reset_command(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    udata.pop('llama_history', None)
    udata.pop('deepseek_history', None)
    send_telegram_message(update, "Your chat history has been reset.")

# Add handler for regular text messages
def message_handler(update, context):
    user_id = update.effective_user.id
    udata = get_user_data(user_id)
    text = update.message.text
    
    # Process based on the button/text content
    if text == "Chat with LLAMA":
        context.args = ["Hi,", "I'd", "like", "to", "chat"]
        return llama_command(update, context)
    elif text == "Chat with Deepseek":
        context.args = ["Hi,", "I'd", "like", "to", "chat"]
        return deepseek_command(update, context)
    elif text == "Predict DBS Price":
        send_telegram_message(update, "Please enter the USD/SGD rate to predict DBS price.\nFormat: 1.34")
        udata['expecting_usdsgd'] = True
        return
    elif text == "Reset Conversation":
        return reset_command(update, context)
    elif udata.get('expecting_usdsgd', False):
        # User is expected to provide USD/SGD rate
        udata['expecting_usdsgd'] = False
        try:
            rate = float(text.strip())
            # Create a context-like object with args
            context.args = [str(rate)]
            return predict_command(update, context)
        except ValueError:
            send_telegram_message(update, "Invalid input. Please provide a valid number for USD/SGD rate.")
            return
    
    # Default: treat as a question for the last used model or LLAMA
    model = udata.get('last_model', 'llama')
    if model == 'deepseek':
        # Set args directly instead of modifying the message text
        context.args = text.split()
        return deepseek_command(update, context)
    else:  # Default to LLAMA
        # Set args directly instead of modifying the message text
        context.args = text.split()
        return llama_command(update, context)

# Register handlers with the dispatcher
telegram_dispatcher.add_handler(CommandHandler("start", start))
telegram_dispatcher.add_handler(CommandHandler("help", help_command))
telegram_dispatcher.add_handler(CommandHandler("llama", llama_command))
telegram_dispatcher.add_handler(CommandHandler("deepseek", deepseek_command))
telegram_dispatcher.add_handler(CommandHandler("predict", predict_command))
telegram_dispatcher.add_handler(CommandHandler("reset", reset_command))
# Add handler for regular text messages (must be added last)
from telegram.ext import MessageHandler, Filters
telegram_dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, message_handler))

@app.route("/telegram_webhook", methods=["POST"])
def telegram_webhook():
    try:
        data = request.get_json(force=True)
        print("Received Telegram update:", data)  # Debug incoming updates
        
        # Check if we have a valid token
        if not TELEGRAM_BOT_TOKEN:
            print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set!")
            return jsonify(success=False, error="Bot token not configured"), 500
            
        update = Update.de_json(data, TELEGRAM_BOT)
        if update:
            print(f"Processing update ID: {update.update_id}, type: {'message' if update.message else 'callback_query' if update.callback_query else 'unknown'}")
            telegram_dispatcher.process_update(update)
        else:
            print("WARNING: Received invalid update format from Telegram")
            
        return jsonify(success=True)
    except Exception as e:
        print(f"ERROR in telegram_webhook: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify(success=False, error=str(e)), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/telegram')
def telegram_info():
    # Check webhook status with Telegram API
    webhook_status = "Unknown"
    try:
        # Get webhook info from Telegram API
        import requests
        
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        if not token:
            webhook_status = "Error: TELEGRAM_BOT_TOKEN not set"
        else:
            response = requests.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
            if response.status_code == 200:
                webhook_info = response.json()
                if webhook_info.get("ok"):
                    webhook_data = webhook_info.get("result", {})
                    if webhook_data.get("url"):
                        webhook_status = "Active: " + webhook_data.get("url")
                        # Check for pending updates
                        pending = webhook_data.get("pending_update_count", 0)
                        if pending > 0:
                            webhook_status += f" ({pending} pending updates)"
                    else:
                        webhook_status = "Not set"
                else:
                    webhook_status = f"Error: {webhook_info.get('description', 'Unknown error')}"
            else:
                webhook_status = f"Error: HTTP {response.status_code}"
    except Exception as e:
        webhook_status = f"Error checking webhook: {str(e)}"
    
    return render_template('telegram.html', 
                          status="GroqSeeker_Bot is ready to use in Telegram", 
                          webhook_status=webhook_status)

@app.route("/main",methods=["GET","POST"])
def main():
    q = request.form.get("q")
    # db
    return(render_template("main.html"))

@app.route("/deepseek",methods=["GET","POST"])
def deepseek():
    return render_template("deepseek.html")

@app.route("/deepseek_reply", methods=["GET", "POST"])
def deepseek_reply():
    q = request.form.get("q")
    client = Groq()
    completion_ds = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": q
           }
        ]
    )
    return(render_template("deepseek_reply.html", r=completion_ds.choices[0].message.content))

@app.route("/llama",methods=["GET","POST"])
def llama():
    return(render_template("llama.html"))

@app.route("/llama_reply",methods=["GET","POST"])
def llama_reply():
    q = request.form.get("q")
    # load model
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return(render_template("llama_reply.html",r=completion.choices[0].message.content))

@app.route("/dbs",methods=["GET","POST"])
def dbs():
    return(render_template("dbs.html"))

@app.route("/prediction",methods=["GET","POST"])
def prediction():
    q = float(request.form.get("q"))
    # Load the trained model
    model = joblib.load("dbs.jl")
    pred_value = round(float(model.predict([[q]])[0]), 2)
    return render_template("prediction.html", r=pred_value)

if __name__ == "__main__":
    app.run()

