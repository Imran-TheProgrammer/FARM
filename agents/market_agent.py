# agents/market_agent.py
from .base import Agent
from typing import Dict, Any, Optional
import requests
from datetime import datetime
import json

class MarketPriceAgent(Agent):
    def __init__(self):
        super().__init__("MarketPriceAgent", [
            "price", "rate", "market", "cost", "rates", "prices",
            "vegetable", "fruit", "commodity", "wholesale", "sabzi mandi"
        ])

    def format_rate_list_for_llm(self, query: str) -> str:
        """Format the rate list data for LLM consumption"""
        formatted_data = """
Date: 12-12-2024
Location: Sabzi Mandi Market, Lahore

Vegetables and Commodities Price List (Rs/Kg):

First Grade Items:
- Bhindi: 42-45
- Baigan: 37-40
- Gobhi: 52-55
- Phool Gobhi: 57-60
- Gandhari Mirch: 90-95
- Shimla Mirch: 85-90
- Tinda: 28-30
- Arvi: 145-150
- Turi: 165-170
- Karela: 105-110
- Sabz Mirch (Green Chili): 85-90
- Kachha Mirch: 67-70
- Surkha: 190-200
- Mooli: 23-25
- Lehsun Patti: 55-60
- Kado Kaddu: 28-30
- Tori: 57-60
- Tinda Desi: 57-60
- Palak: 28-30
- Methi & Dhania: 42-45
- Ginger: 95-100
- Thai Ginger: 28-30
- Patta: 47-50
- Khatta Podina: 140-145
- Desi Gajar: 67-70
- Patta Mooli: 37-40
- Patta Pyaz: 95-100

Major Commodities:
- Aloo (Potato) First: 85-90 (50kg)
- Aloo Second: 75-80 (50kg)
- Aloo Third: 65-70 (50kg)
- Pyaz (Onion) First: 127-135 (110kg)
- Pyaz Second: 118-125 (70kg)
- Pyaz Third: 105-110 (70kg)
- Tamatar (Tomato) First: 104-110 (80kg)
- Tamatar Second: 90-95 (50kg)
- Tamatar Third: 75-80 (50kg)
- Lehsan Desi: 525-545
- Lehsan Special: 580-600
- Lehsan G1: 325-340
- Adrak Thailand: 375-390
- Adrak China: 375-390
- Kheera Factory: 28-30

Note: All prices are in PKR (Pakistani Rupees)
Complaint Number: 0800-02345
"""
        return formatted_data

    async def get_llm_response(self, client, query: str, market_data: str) -> str:
        """Get LLM analysis of market data"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are an expert market analyst for the Lahore Sabzi Mandi. 
                    Analyze the provided price list and answer queries about vegetable and commodity prices. 
                    Consider:
                    1. Current prices and ranges
                    2. Price differences between grades
                    3. Wholesale vs. retail implications
                    4. Market trends and patterns
                    5. Recommendations for buyers and sellers
                    
                    Provide specific prices when asked and offer practical insights for farmers and traders."""},
                    {"role": "user", "content": f"""
                    User Query: {query}
                    
                    Current Market Rates:
                    {market_data}
                    
                    Please provide a detailed response focusing on the relevant items and their prices."""}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return "I apologize, but I'm having trouble analyzing the market data at the moment."

    async def execute(self, query: str, openai_client) -> str:
        """Execute market price analysis"""
        try:
            # Format the rate list data
            formatted_data = self.format_rate_list_for_llm(query)
            
            # Get LLM analysis
            response = await self.get_llm_response(openai_client, query, formatted_data)
            
            return response

        except Exception as e:
            print(f"Error in market price agent execution: {str(e)}")
            return "I encountered an error while analyzing the market prices. Please try again later."