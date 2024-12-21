# agents/controller.py
from typing import List, Dict, Optional
import openai
from datetime import datetime
from pathlib import Path
import aiofiles
from .base import Agent
from .disease_agent import DiseaseDetectionAgent
from .weather_agent import WeatherAgent
from .market_agent import MarketPriceAgent

class AgentController:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        # Initialize only required agents
        self.agents = [
            DiseaseDetectionAgent(),
            WeatherAgent(),
            MarketPriceAgent()
        ]
        self.chat_history = []
        self.audio_folder = Path("static/audio")
        self.audio_folder.mkdir(parents=True, exist_ok=True)
        self.voice_mode_enabled = False

    async def generate_speech(self, text: str) -> Optional[str]:
        """Generate speech from text using OpenAI TTS"""
        try:
            speech_file_path = self.audio_folder / f"response_{len(self.chat_history)}.mp3"
            
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )

            async with aiofiles.open(str(speech_file_path), 'wb') as file:
                await file.write(response.content)

            return f"/static/audio/{speech_file_path.name}"
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None
        

    def is_market_query(self, query: str) -> bool:
        """Check if query is market-related"""
        market_keywords = [
            "price", "rate", "market", "cost", "rates", "prices",
            "vegetable", "fruit", "commodity", "wholesale"
        ]
        return any(keyword in query.lower() for keyword in market_keywords)
    
                   
    def format_messages_for_gpt(self) -> List[Dict]:
        """Format chat history for GPT context"""
        messages = [{
            "role": "system",
            "content": """You are F.A.R.M (Future of Agriculture with Revolutionary Model), 
            an AI assistant specialized in agriculture. Provide expert guidance on:
            - Crop management and cultivation
            - Pest control and disease prevention
            - Soil health and fertility
            - Sustainable farming practices
            - Modern agricultural techniques
            
            Maintain context from previous messages and provide detailed, actionable advice."""
        }]
        
        # Add last 10 messages for context
        for msg in self.chat_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return messages

    def get_gpt_response(self, query: str) -> str:
        """Get a contextual response from GPT"""
        try:
            messages = self.format_messages_for_gpt()
            messages.append({"role": "user", "content": query})
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I'm having trouble responding. Error: {str(e)}"

    def is_weather_query(self, query: str) -> bool:
        """Check if query is weather-related"""
        weather_keywords = [
            "weather", "temperature", "rain", "rainfall", "humidity",
            "forecast", "climate", "precipitation", "irrigation",
            "sunny", "rainy", "cloudy", "storm"
        ]
        return any(keyword in query.lower() for keyword in weather_keywords)

    def extract_location(self, query: str) -> str:
        """Extract location from query using LLM"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract the location mentioned in the query. If no location is mentioned, respond with 'unknown'."},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Error extracting location: {str(e)}")
            return "unknown"

    async def process_query(self, query: str, image_data: bytes = None, voice_mode: bool = False) -> Dict:
        timestamp = datetime.now().strftime("%H:%M")
        
        try:
            if image_data:
                # Handle disease detection
                disease_agent = next(
                    (agent for agent in self.agents if isinstance(agent, DiseaseDetectionAgent)), 
                    None
                )
                if disease_agent:
                    response = disease_agent.execute(image_data)
                else:
                    response = "Disease detection service is currently unavailable."
            
            elif self.is_weather_query(query):
                # Handle weather query
                weather_agent = next(
                    (agent for agent in self.agents if isinstance(agent, WeatherAgent)), 
                    None
                )
                if weather_agent:
                    location = self.extract_location(query)
                    if location == "unknown":
                        response = "I noticed you're asking about weather, but I couldn't determine the location. Could you please specify the location you're interested in?"
                    else:
                        response = await weather_agent.execute(query, location, self.client)
                else:
                    response = "Weather service is currently unavailable."
            
            elif self.is_market_query(query):
                # Handle market price query
                market_agent = next(
                    (agent for agent in self.agents if isinstance(agent, MarketPriceAgent)), 
                    None
                )
                if market_agent:
                    response = await market_agent.execute(query, self.client)
                else:
                    response = "Market price service is currently unavailable."
            
            else:
                # Handle general queries
                response = self.get_gpt_response(query)

            # Generate audio if voice mode is enabled
            audio_url = await self.generate_speech(response) if voice_mode else None

            # Add assistant response to history
            self.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": timestamp,
                "audio_url": audio_url,
                "voice_mode": voice_mode
            })

            return {
                "response": response,
                "audio_url": audio_url,
                "history": self.chat_history,
                "voice_mode": voice_mode
            }

        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return {
                "error": str(e),
                "history": self.chat_history
            }