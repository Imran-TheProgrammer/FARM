# agents/weather_agent.py
from .base import Agent
from typing import Dict, Any, Optional
import requests
import json
import os
from datetime import datetime

class WeatherAgent(Agent):
    def __init__(self):
        super().__init__("WeatherAgent", [
            "weather", "temperature", "rain", "rainfall", "humidity", 
            "forecast", "climate", "precipitation", "irrigation"
        ])
        self.api_key = os.getenv('ACCUWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("AccuWeather API key not found in environment variables")
        
        self.base_url = "http://dataservice.accuweather.com"
        
    def get_location_key(self, location: str) -> Optional[str]:
        """Get AccuWeather location key from city name"""
        try:
            url = f"{self.base_url}/locations/v1/cities/search"
            params = {
                "apikey": self.api_key,
                "q": location,
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            locations = response.json()
            if locations:
                return locations[0]["Key"]
            return None
            
        except Exception as e:
            print(f"Error getting location key: {str(e)}")
            return None

    def get_current_conditions(self, location_key: str) -> Optional[Dict]:
        """Get current weather conditions"""
        try:
            url = f"{self.base_url}/currentconditions/v1/{location_key}"
            params = {
                "apikey": self.api_key,
                "details": True
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()[0]
            
        except Exception as e:
            print(f"Error getting current conditions: {str(e)}")
            return None

    def get_weather_forecast(self, location_key: str) -> Optional[Dict]:
        """Get 5-day weather forecast"""
        try:
            url = f"{self.base_url}/forecasts/v1/daily/5day/{location_key}"
            params = {
                "apikey": self.api_key,
                "details": True,
                "metric": True
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error getting forecast: {str(e)}")
            return None

    def format_conditions_for_llm(self, conditions: Dict, forecast: Dict) -> str:
        """Format weather data for LLM input"""
        data = {
            "current": {
                "temperature": conditions.get("Temperature", {}).get("Metric", {}).get("Value"),
                "humidity": conditions.get("RelativeHumidity"),
                "precipitation": conditions.get("HasPrecipitation"),
                "uv_index": conditions.get("UVIndex"),
                "wind_speed": conditions.get("Wind", {}).get("Speed", {}).get("Metric", {}).get("Value")
            },
            "forecast": [
                {
                    "date": day.get("Date"),
                    "min_temp": day.get("Temperature", {}).get("Minimum", {}).get("Value"),
                    "max_temp": day.get("Temperature", {}).get("Maximum", {}).get("Value"),
                    "day_forecast": day.get("Day", {}).get("IconPhrase"),
                    "rain_probability": day.get("Day", {}).get("RainProbability")
                }
                for day in forecast.get("DailyForecasts", [])
            ]
        }
        return json.dumps(data, indent=2)

    def get_llm_response(self, client, query: str, weather_data: str) -> str:
        """Get LLM interpretation of weather data"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are an agricultural weather expert. 
                    Analyze the weather data and provide farming-related insights and recommendations.
                    Focus on how the weather conditions affect farming activities, irrigation needs,
                    and potential risks or opportunities for crops."""},
                    {"role": "user", "content": f"""
                    Query: {query}
                    Weather Data: {weather_data}
                    
                    Provide a detailed analysis focusing on agricultural implications."""}
                ]
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return None

    async def execute(self, query: str, location: str, openai_client) -> str:
        """Execute weather analysis for query"""
        try:
            # Get location key
            location_key = self.get_location_key(location)
            if not location_key:
                return "I'm sorry, but I couldn't find that location. Please check the spelling and try again."

            # Get weather data
            current_conditions = self.get_current_conditions(location_key)
            forecast = self.get_weather_forecast(location_key)
            
            if not current_conditions or not forecast:
                return "I'm sorry, but I couldn't fetch the weather data at the moment. Please try again later."

            # Format weather data for LLM
            weather_data = self.format_conditions_for_llm(current_conditions, forecast)
            
            # Get LLM analysis
            response = self.get_llm_response(openai_client, query, weather_data)
            
            if not response:
                return "I apologize, but I couldn't analyze the weather data at the moment. Please try again."

            return response

        except Exception as e:
            print(f"Error in weather agent execution: {str(e)}")
            return "I encountered an error while analyzing the weather data. Please try again later."