# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
import os
from agents.controller import AgentController
import base64

# Load environment variables
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")

# Initialize Flask app with specific settings
app = Flask(__name__, 
            static_folder='static', 
            static_url_path='/static')

# Initialize controller outside of route handlers
controller = AgentController(OPENAI_API_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
async def process_query():
    try:
        data = request.json
        image_data = None
        
        # Check if image data is included in the request
        if 'image' in data:
            image_data = base64.b64decode(data['image'].split(',')[1])
        
        result = await controller.process_query(
            data.get('query', ''),
            image_data=image_data,
            voice_mode=data.get('voice_mode', False)
        )
        return jsonify(result)
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)

if __name__ == '__main__':
    # Run without debug mode and with threaded=False
    app.run(host='127.0.0.1', 
            port=5000, 
            debug=False, 
            use_reloader=False, 
            threaded=False)