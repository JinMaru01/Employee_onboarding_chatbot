import re
from difflib import get_close_matches

def respond_to_mission_question(intent: str, question: str, mission_data: dict) -> str:

    message = mission_data.get("message", "The mission of the company is")
    missions = mission_data.get("response", {})
    mission_titles = list(missions.keys())
    user_question = question.lower()

    match = re.search(r"mission\s*(\d+)", user_question)
    if match:
        mission_index = int(match.group(1)) - 1
        if 0 <= mission_index < len(mission_titles):
            selected_title = mission_titles[mission_index]
            return missions[selected_title]

    # Match based on mission title keywords
    matched = [title for title in mission_titles if title.lower() in user_question]
    if len(matched) == 1:
        return missions[matched[0]]

    # Check for request to show all missions
    if any(phrase in user_question for phrase in ["all missions", "details of all", "each mission", "full mission"]):
        return "\n\n".join([f"**{title}**: {desc}" for title, desc in missions.items()])

    # General intent: list all mission titles
    if intent == "ask_for_mission":
        listed_titles = ", ".join(mission_titles)
        return f"{message} {listed_titles}."

    # Fallback
    return "Could you please clarify which mission you're referring to?"

mission_aliases = {
    "customer": "Customer at the heart",
    "community": "Community as the cause",
    "employee": "Employee as the pillar",
    "shareholder": "Deliver shareholder value",
    "mission 1": "Customer at the heart",
    "mission 2": "Community as the cause",
    "mission 3": "Employee as the pillar",
    "mission 4": "Deliver shareholder value",
    "first mission": "Customer at the heart",
    "second mission": "Community as the cause",
    "third mission": "Employee as the pillar",
    "fourth mission": "Deliver shareholder value"
}

mission_data = {
    "message": "The mission of the company is",
    "response": {
        "Customer at the heart": "To engage and understand customer need, provide best-in-class product and Services, be respective and quick in resolving-resulting in true customer delight.",
        "Community as the cause": "To deliver robust and cost-effective mobile money service that promote financial inclusion, catalyze growth, and reduce social inequality.",
        "Employee as the pillar": "To provide an enabling work culture, where career aspirations can be realized through consistent performance and demonstration of WING's core values.",
        "Deliver shareholder value": "To demonstrate strong corporate governance standards that protects and balance shareholder interests in the journey to achieving short and long-term business goals and staying relevant by being the first bank to offer innovate mobile financial service, accessible to all."
    }
}

def respond_to_mission_question(intent: str, question: str, mission_data: dict) -> str:
    message = mission_data.get("message", "The mission of the company is")
    missions = mission_data.get("response", {})
    mission_titles = list(missions.keys())
    user_question = question.lower()
        
    match = re.search(r"mission\s*(\d+)", user_question)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(mission_titles):
            return missions[mission_titles[idx]]
        else:
            return f"There is no mission number {match.group(1)}. Please ask about mission 1 to {len(mission_titles)}."

    # Alias keyword matching
    for keyword, title in mission_aliases.items():
        if keyword in user_question:
            return missions.get(title)

    # Fuzzy match using difflib
    for title in mission_titles:
        if get_close_matches(title.lower(), [user_question], n=1, cutoff=0.6):
            return missions[title]

    # Asking for all missions
    if any(p in user_question for p in ["all missions", "each mission", "full list"]):
        return "\n\n".join([f"**{t}**: {d}" for t, d in missions.items()])

    # General intent fallback
    if intent == "ask_for_mission":
        return f"{message} {', '.join(mission_titles)}."

    return "Could you clarify which mission you're referring to?"
