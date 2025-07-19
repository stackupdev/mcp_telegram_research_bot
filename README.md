# DBS Prediction Web Service

This is a Flask-based web application for making predictions using a trained model (`dbs.jl`) and interacting with the Groq API (LLM). The app features a simple web interface for users to input data, get model predictions, and interact with an LLM via the Groq API.

## Features

- **Prediction Endpoint:** Enter a value and receive a model prediction.
- **LLM Integration:** Ask questions and get responses from the Groq Llama model.
- **Deepseek Chatbot:** Interact with the Deepseek model via the Groq API.
- **Ready for Cloud Deployment:** Designed for easy deployment on Render.com.

## How the App Works

### User Experience Flow

1. **Landing Page (`/`)**: The user is greeted and asked to enter their name.
2. **Main Menu (`/main`)**: After submitting their name, the user chooses between three options:
    - **LLAMA Chatbot**: Interact with an AI chatbot powered by Groq's Llama model.
    - **Deepseek Chatbot**: Interact with the Deepseek model via Groq API.
    - **DBS Prediction**: Predict the DBS share price based on the USD/SGD exchange rate.

#### LLAMA Chatbot Flow
- The user selects the LLAMA chatbot option and is taken to `/llama`, where they can input a question or prompt.
- On submission, the app sends the query to the Groq Llama model via the Groq API, receives a response, and displays it on `/llama_reply`.
- The user can return to the main menu from here.

#### Deepseek Chatbot Flow
- The user selects the Deepseek chatbot option and is taken to `/deepseek`, where they can input a question or prompt.
- On submission, the app sends the query to the Deepseek model via the Groq API, receives a response, and displays it on `/deepseek_reply`.
- The user can return to the main menu from here.

#### DBS Prediction Flow
- The user selects the prediction option and is taken to `/dbs`, where they input the USD/SGD exchange rate.
- On submission, the app loads the trained model (`dbs.jl`), makes a prediction, and displays the predicted DBS share price on `/prediction`.
- The user can return to the home page from here.

### Data Flow
- **User inputs** (name, chatbot query, exchange rate) are submitted via HTML forms.
- Flask routes handle the form data:
    - For chatbots: The query is sent to the Groq API (LLAMA or Deepseek), and the response is passed to the template.
    - For prediction: The exchange rate is passed to the model, and the prediction is rendered in the template.
- **Templates** display the results and guide users through the next steps.

### Summary Table

| Route            | Template             | User Action / Purpose                        | Data Flow                                   |
|------------------|----------------------|----------------------------------------------|---------------------------------------------|
| `/`              | index.html           | Enter name                                   | Form input → `/main`                        |
| `/main`          | main.html            | Choose Chatbot or Prediction                 | Button → `/llama`, `/deepseek`, or `/dbs`   |
| `/llama`         | llama.html           | Enter LLAMA chatbot query                    | Form input → `/llama_reply`                 |
| `/llama_reply`   | llama_reply.html     | View LLAMA chatbot response                  | Query sent to Groq API, response displayed  |
| `/deepseek`      | deepseek.html        | Enter Deepseek chatbot query                 | Form input → `/deepseek_reply`              |
| `/deepseek_reply`| deepseek_reply.html  | View Deepseek chatbot response               | Query sent to Groq API, response displayed  |
| `/dbs`           | dbs.html             | Enter USD/SGD exchange rate                  | Form input → `/prediction`                  |
| `/prediction`    | prediction.html      | View predicted DBS share price               | Model loaded, prediction displayed          |

## Requirements

- Python 3.11
- gunicorn
- flask
- joblib
- scikit-learn
- groq

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### Deployment on Render.com

1. **Connect your repository** to Render.com and create a new Web Service.
2. **Set the environment variable** in the Render dashboard:
    - Key: `GROQ_API_KEY`
    - Value: _your actual Groq API key_
3. **Ensure your `dbs.jl` model file** is present in the root directory of the repo.
4. Render will install dependencies from `requirements.txt` and run your app.

## Project Structure

```
dbs_pred/
├── app.py
├── dbs.jl
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── main.html
│   ├── llama.html
│   ├── llama_reply.html
│   ├── deepseek.html
│   ├── deepseek_reply.html
│   ├── dbs.html
│   └── prediction.html
└── README.md
```

## API Endpoints

- `/` — Home page
- `/main` — Main interface
- `/llama` — LLM input page
- `/llama_reply` — LLM response page
- `/dbs` — DBS info page
- `/prediction` — Model prediction page
- `/deepseek` — Deepseek chatbot page
- `/deepseek_reply` — Deepseek chatbot response page

## Security

- **Do NOT hardcode your API keys** in the codebase.
- Set `GROQ_API_KEY` as an environment variable, especially for cloud deployments.

## License

[CLARENCE](LICENSE) (or your preferred license)