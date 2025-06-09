import logging
import os
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

def register_tools(mcp):
    @mcp.tool()
    async def city_weather(city: str) -> str:
        """Fetch current weather for a given city using the OpenWeather API."""
        logger.info(f"Fetching weather for city: {city}")
        try:
            api_key = os.getenv("OPENWEATHER_API_KEY")
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            res = requests.get(url).json()
            if "weather" not in res or "main" not in res:
                return f"Could not retrieve weather for '{city}'."
            desc = res['weather'][0]['description']
            temp = res['main']['temp']
            return f"The weather in {city.title()} is {desc} with a temperature of {temp:.1f}°C."
        except Exception as e:
            return f"Couldn't fetch weather: {e}"
