import time
import logging
import weave
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary
from agent.system_prompt import system_prompt

# Set up module logging
logger = logging.getLogger(__name__)

@weave.op()
def perception_step(frame, state_data, vlm):
    """
    Observe and describe your current situation using both visual and comprehensive state data.
    Returns (observation, slow_thinking_needed)
    """
    # Format the comprehensive state context using the utility
    state_context = format_state_for_llm(state_data)
    
    # Log the state data being used
    state_summary = format_state_summary(state_data)
    logger.info("[PERCEPTION] Processing frame with comprehensive state data")
    logger.info(f"[PERCEPTION] State: {state_summary}")
    logger.info(f"[PERCEPTION] State context length: {len(state_context)} characters")
    
    perception_prompt = f"""You are the agent, actively playing Pokemon Emerald. Observe and describe your current situation in detail. Use both the visual frame and the comprehensive game state data:
<game_state>
{state_context}
</game_state>

Based on the visual frame and the above game state data, describe your current situation:
- CUTSCENE or TITLE SCREEN: What does the cutscene or title screen show?
- MAP: You are navigating a terrain (city, forest, grassland, etc.). Are there any interactable locations (NPCs, items, doors)? What are the traversable vs. non-traversable areas? Use your position coordinates to understand where you are.
- BATTLE: Analyze the battle situation using both visual and state data. What moves are available? What's the strategy?
- DIALOGUE: What is the character telling you? How important is this information? Can you respond to the NPC?
- MENU: What menu are you in? What options are available? What should you select based on your current needs?
    
Combine visual observation with the game state data to give a complete picture of the current situation.
You should mostly carefully describe the current and immediate observation in detail, so the next action step can make a good decision."""
    
    observation = vlm.get_query(frame, system_prompt + perception_prompt, "PERCEPTION")

    return observation 