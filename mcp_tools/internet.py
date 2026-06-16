import json
import urllib.request
import urllib.parse

def get_current_weather(location: str) -> str:
    """
    Fetches the current real-time weather conditions for any location or city 
    using a free public weather service. No API key required.
    Args:
        location (str): The name of the city or location (e.g., "Paris", "New York").
    Returns:
        str: Current weather conditions.
    """
    try:
        # Encode the location name safely
        encoded_loc = urllib.parse.quote(location)
        # Fetch weather from free wttr.in in a simplified text format
        url = f"https://wttr.in/{encoded_loc}?format=%l:+%C+%t+Humidity:+%h+Wind:+%w"
        # Crucial: Use 'curl' as the User-Agent so wttr.in returns raw text instead of a massive HTML page
        req = urllib.request.Request(url, headers={'User-Agent': 'curl'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read().decode('utf-8').strip()
    except Exception as e:
        return f"Error fetching weather for '{location}': {e}"

def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia and retrieves a clean text summary of any topic, person, 
    event, or concept. Useful for looking up general knowledge. No API key required.
    Args:
        query (str): The topic or search term to look up on Wikipedia.
    Returns:
        str: A summary of the search topic.
    """
    try:
        # Step 1: Search for the best matching page title
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded_query}&limit=1&namespace=0&format=json"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            if not data or len(data) < 2 or not data[1]:
                return f"No Wikipedia match found for '{query}'."
            best_title = data[1][0]
        
        # Step 2: Fetch the plain text summary for that title
        encoded_title = urllib.parse.quote(best_title)
        summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={encoded_title}&format=json"
        req_summary = urllib.request.Request(summary_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req_summary, timeout=5) as response:
            summary_data = json.loads(response.read().decode('utf-8'))
            pages = summary_data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if "extract" in page_data:
                    return page_data["extract"]
            return f"Could not retrieve summary for '{best_title}'."
    except Exception as e:
        return f"Error searching Wikipedia for '{query}': {e}"
