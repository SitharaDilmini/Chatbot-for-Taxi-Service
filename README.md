# Taxi Service Chatbot

This project is a supervised learning chatbot designed for a taxi service. The chatbot helps users to get information about the taxi service. It is trained using TensorFlow and can be customized to respond to different intents by modifying the **intents.json** file.    

# How It Works
1. **Input Training Data**  
   	- Define all the questions and responses as done in the `intents.json`.

2. **Model Training**  
    - Run `model.py` to train the model on data in the `intents.json`.
    - A new model will be trained and saved.

## Getting Started
### **Backend Setup**
1. Ensure following are installed.
   	- Node.js (for Angular):  [Download Node.js](https://nodejs.org/en)
   	- Python 3.8+:   [Download Python](https://www.python.org/)
3. Clone the repository
4. Navigate to the backend directory
	`cd backend`
5. Create and activate a virtual environment
	- **windows :**
		- `python -m venv env`
		- `env\Scripts\activate`
	- **macOS/Linux :**
		- `python -m venv env`
		- `source env/bin/activate`
6. Install dependencies
	`pip install -r requirements.txt`
7. Train the model
	`python model.py`
8. Run the backend server
	`python app.py`

### **Frontend Setup**
1. Navigate to the frontend directory:
	`cd frontend/chatbot-frontend`
2. Install dependencies
	`npm install`
3. Start the Angular development server
	`ng serve`
