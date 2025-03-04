import random

def impromptu_speaking():
    
    """Give a random topic and evaluate user response."""
    topics = [
        "Why is empathy important?",
        "The impact of technology on daily life.",
        "Should schools focus more on creativity?"
    ]
    topic = random.choice(topics)
    
    return f"Your topic is {topic}. Speak about it."

def storytelling():
    """Ask the user to tell a short story and analyze its structure."""
    return "Tell a short story about a moment that changed your life."

def conflict_resolution():
    """Provide a conflict scenario and evaluate user diplomacy."""
    scenarios = [
        "Your teammate missed a deadline. How do you address it?",
        "A colleague disagrees with your idea in a meeting. How do you respond?",
        "A friend is upset with you. How do you handle the situation?"
    ]
    scenario = random.choice(scenarios)
    
    return f"Scenario: {scenario} - Respond appropriately."

def start_training(module_type):
    """Selects and runs a training module based on user choice."""
    if module_type == "Impromptu Speaking":
        return impromptu_speaking()
    elif module_type == "Storytelling":
        return storytelling()
    elif module_type == "Conflict Resolution":
        return conflict_resolution()
    else:
        return "Invalid module selection."