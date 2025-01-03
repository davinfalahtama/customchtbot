# Custom Indonesia AI Chatbot

A custom chatbot built with Streamlit that helps users learn more about Indonesia AI company, its services, and programs. The chatbot uses OpenAI's GPT model and Pinecone vector database for intelligent responses.

## Features

- Interactive chat interface
- Context-aware responses about Indonesia AI
- Access control middleware
- Responsive UI with WhatsApp-like chat bubbles
- Vector database integration for accurate information retrieval

## Prerequisites

Before running this application, make sure you have:

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment
- Git (for deployment)

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Usage

To run the application locally:

```bash
streamlit run main.py
```

Access the application at: `http://localhost:8501?access=indo-ai`

Note: The application requires the correct access parameter in the URL to function.

## Deployment on Streamlit Cloud

1. Push your code to GitHub:

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)

3. Click on "New app" and select your repository

4. Configure the deployment:

   - Set Python version (3.8+)
   - Add your environment variables (OPENAI_API_KEY, etc.)
   - Set the main file path as `main.py`

5. Click "Deploy"

### Important Deployment Notes

- Ensure all requirements are listed in `requirements.txt`
- Securely store API keys in Streamlit Cloud's secrets management
- Configure proper access control parameters

## Security

- The application implements middleware access control
- API keys should never be committed to the repository
- Use environment variables for sensitive information

## Add Context

- Only run `add_context.py` when it's an update from dataset.

```bash
python add_context.py
```
