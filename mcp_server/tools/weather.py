import logging
from mcp_server.utils.http import make_nws_request

logger = logging.getLogger("weather_tools")

def register_tools(mcp):
    @mcp.tool()
    async def get_alerts(state: str) -> str:
        logger.info(f"Tool called: get_alerts({state=})")
        try:
            url = f"https://api.weather.gov/alerts/active/area/{state}"
            data = await make_nws_request(url)
            if not data or "features" not in data:
                return "No alert data found or unable to fetch."
            if not data["features"]:
                return "No active alerts for this state."
            return "\n---\n".join(format_alert(f) for f in data["features"])
        except Exception:
            logger.exception("Error in get_alerts")
            return "Error retrieving alerts."

    @mcp.tool()
    async def get_forecast(latitude: float, longitude: float) -> str:
        logger.info(f"Tool called: get_forecast({latitude=}, {longitude=})")
        try:
            points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
            points_data = await make_nws_request(points_url)
            forecast_url = points_data.get("properties", {}).get("forecast", "")
            if not forecast_url:
                return "Could not determine forecast URL."
            forecast_data = await make_nws_request(forecast_url)
            periods = forecast_data.get("properties", {}).get("periods", [])
            return "\n---\n".join(
                f"{p.get('name')}:\nTemp: {p.get('temperature')}°{p.get('temperatureUnit')}\n"
                f"Wind: {p.get('windSpeed')} {p.get('windDirection')}\nForecast: {p.get('detailedForecast')}"
                for p in periods[:5]
            )
        except Exception:
            logger.exception("Error in get_forecast")
            return "Error retrieving forecast."

def format_alert(feature: dict) -> str:
    props = feature.get("properties", {})
    return f"""Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No instructions provided')}"""
