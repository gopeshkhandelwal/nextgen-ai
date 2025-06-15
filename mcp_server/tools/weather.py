import logging
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

def register_tools(mcp):
    """
    Register tool functions with the MCP (Model Context Protocol) client.
    """

    @mcp.tool()
    async def city_weather(city: str) -> str:
        """
        Fetch current weather for a given city using the OpenWeatherMap API.

        Args:
            city (str): Name of the city to fetch weather for.

        Returns:
            str: Description of the weather or an error message.
        """
        logger.info("Fetching weather for city: %s", city)

        try:
            api_key = os.getenv("OPENWEATHER_API_KEY")
            if not api_key:
                logger.error("OPENWEATHER_API_KEY not found in environment.")
                return "Weather API key is not configured."

            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "weather" not in data or "main" not in data:
                logger.warning("Incomplete weather data received for city: %s", city)
                return f"Could not retrieve weather for '{city}'."

            description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            return f"The weather in {city.title()} is {description} with a temperature of {temperature:.1f}°C."

        except requests.RequestException as req_err:
            logger.exception("Request error while fetching weather.")
            return f"Failed to fetch weather data: {req_err}"
        except Exception as e:
            logger.exception("Unexpected error while fetching weather.")
            return f"Couldn't fetch weather: {e}"