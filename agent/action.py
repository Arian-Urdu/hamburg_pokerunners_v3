import logging
import random
import sys
import weave
from agent.system_prompt import system_prompt
from utils.state_formatter import format_state_for_llm, format_state_summary, get_movement_options, get_party_health_summary
from utils.vlm import VLM

# Set up module logging
logger = logging.getLogger(__name__)

@weave.op()
def action_step(memory_context, current_plan, latest_observation, state_data, recent_actions, vlm):
    """
    Decide and perform the next action button(s) based on memory, plan, observation, and comprehensive state.
    Returns a list of action buttons as strings.
    """
    # Get formatted state context and useful summaries
    state_context = format_state_for_llm(state_data)
    state_summary = format_state_summary(state_data)
    movement_options = get_movement_options(state_data)
    party_health = get_party_health_summary(state_data)
    
    logger.info("[ACTION] Starting action decision")
    logger.info(f"[ACTION] State: {state_summary}")
    logger.info(f"[ACTION] Party health: {party_health['healthy_count']}/{party_health['total_count']} healthy")
    if movement_options:
        logger.info(f"[ACTION] Movement options: {movement_options}")
    
    # Build enhanced action context
    action_context = []
    
    # Extract key info for context
    game_data = state_data.get('game', {})
    
    # Battle vs Overworld context
    if game_data.get('in_battle', False):
        action_context.append("=== BATTLE MODE ===")
        battle_info = game_data.get('battle_info', {})
        if battle_info:
            if 'player_pokemon' in battle_info:
                player_pkmn = battle_info['player_pokemon']
                action_context.append(f"Your Pokemon: {player_pkmn.get('species_name', player_pkmn.get('species', 'Unknown'))} (Lv.{player_pkmn.get('level', '?')}) HP: {player_pkmn.get('current_hp', '?')}/{player_pkmn.get('max_hp', '?')}")
            if 'opponent_pokemon' in battle_info:
                opp_pkmn = battle_info['opponent_pokemon']
                action_context.append(f"Opponent: {opp_pkmn.get('species_name', opp_pkmn.get('species', 'Unknown'))} (Lv.{opp_pkmn.get('level', '?')}) HP: {opp_pkmn.get('current_hp', '?')}/{opp_pkmn.get('max_hp', '?')}")
    else:
        action_context.append("=== OVERWORLD MODE ===")
        
        # Movement options from utility
        if movement_options:
            action_context.append("Movement Options:")
            for direction, description in movement_options.items():
                action_context.append(f"  {direction}: {description}")
    
    # Party health summary
    if party_health['total_count'] > 0:
        action_context.append("=== PARTY STATUS ===")
        action_context.append(f"Healthy Pokemon: {party_health['healthy_count']}/{party_health['total_count']}")
        if party_health['critical_pokemon']:
            action_context.append("Critical Pokemon:")
            for critical in party_health['critical_pokemon']:
                action_context.append(f"  {critical}")
    
    # Recent actions context
    if recent_actions:
        # Ensure all recent actions are strings before joining
        recent_actions_str = [str(action) for action in recent_actions[-5:]]
        action_context.append(f"Recent Actions: {', '.join(recent_actions_str)}")

    context_str = "\n".join(action_context)
    
    action_prompt = f"""ACTION DECISION TASK
You are the agent playing Pokemon Emerald with a speedrunning mindset. Make quick, efficient decisions.
    
You will be provided with the following information:
<context>
    <game_state>
    {state_context}
    </game_state>

    <action_context>
    {context_str}
    </action_context>

    <memory_context>
    {memory_context}
    </memory_context>

    <current_plan>
    {current_plan if current_plan else 'No plan yet'}
    </current_plan>

    <latest_observation>
    {latest_observation}
    </latest_observation>
</context>

Based on the comprehensive state information above, decide your next action(s):
    
BATTLE STRATEGY:
- If in battle: Choose moves strategically based on type effectiveness and damage
- Consider switching pokemon if current one is weak/low HP
- Use items if pokemon is in critical condition
    
NAVIGATION STRATEGY:
- Use movement options analysis above for efficient navigation
- Avoid blocked tiles (marked as BLOCKED)
- Consider tall grass: avoid if party is weak, seek if need to train/catch
- Navigate around water unless you have Surf
- Use coordinates to track progress toward objectives
    
MENU/DIALOGUE STRATEGY:
- If in dialogue: use A to initiate dialogue, but B to advance wherever possible to not accidentally trigger the same dialogue again even though it was finished
- If in menu: Navigate with UP/DOWN/LEFT/RIGHT, A to select, B to cancel/back out
- If stuck in menu/interface: B repeatedly to exit to overworld
- In Pokemon Center: A to talk to Nurse Joy, A to confirm healing
    
HEALTH MANAGEMENT:
- If pokemon are low HP/fainted, head to Pokemon Center
- If no healthy pokemon, prioritize healing immediately
- Consider terrain: avoid wild encounters if party is weak
    
EFFICIENCY RULES:
1. Output sequences of actions when you know what's coming (e.g., "RIGHT, RIGHT, RIGHT, A" to enter a door)
2. For movement: repeat directions based on movement options (e.g., "UP, UP, UP, UP" if UP shows "Normal path")
3. If uncertain, output single action and reassess
4. Use traversability data: move toward open paths, avoid obstacles

In the overworld focus on using directions to navigate efficiently.
    
Valid buttons: A, B, UP, DOWN, LEFT, RIGHT, START
- A: Interact with NPCs/objects, confirm selections, advance dialogue, use moves in battle
- B: Cancel menus, back out of interfaces, flee from battle
- UP/DOWN/LEFT/RIGHT: Move character, navigate menus (also in battles), select options
- START: Open main menu (Title sequence, Pokedex, Pokemon, Bag, etc.)

MOST IMPORTANT: look at IMMEDIATE NEXT GOAL in the plan and try to achieve it with specific actions.

⚠️ CRITICAL WARNING: NEVER save the game using the in-game save menu! Saving will crash the entire run and end your progress. If you encounter a save prompt in the game, press B to cancel it immediately!
    
Output: 
1. Should be your EXTREMELY DETAILED STEP-BY-STEP  reasoning of how to achieve the IMMEDIATE NEXT GOAL stated in the Plan. In <REASONING> </REASONING> tags.
2. Should include specific actions to take based on the current game state and objectives. In <ACTIONS> </ACTIONS> tags  Return ONLY the button name(s) as a comma-separated list, nothing else. Maximum 5 actions in sequence.

Example output:
<REASONING> To achieve the immediate next goal of entering the building, I will move RIGHT until I reach the door, then press A to enter. </REASONING>
<ACTIONS> RIGHT, RIGHT, RIGHT, A </ACTIONS>

When traversing give multiple movement commands in sequence to move efficiently.
REMEMBER MOST IMPORTANT OF ALL: ACTIONS MUST BE IN <ACTIONS> </ACTIONS> TAGS AND VALID BUTTONS ONLY: A, B, UP, DOWN, LEFT, RIGHT, START"""
    
    # Construct complete prompt for VLM
    complete_prompt = system_prompt + action_prompt
    
    action_response = vlm.get_text_query(complete_prompt, "ACTION").strip().upper()
    valid_buttons = ['A', 'B', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    
    # Print VLM response for debugging
    print("🤖 VLM RESPONSE:")
    print(f"Raw response: '{action_response}'")
    
    # Parse actions from between <ACTIONS> and </ACTIONS> tags
    actions = []
    try:
        # Find the content between <ACTIONS> and </ACTIONS> tags
        start_tag = "<ACTIONS>"
        end_tag = "</ACTIONS>"
        start_idx = action_response.find(start_tag)
        end_idx = action_response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            actions_content = action_response[start_idx + len(start_tag):end_idx].strip()
            # Split by commas and clean up
            actions = [btn.strip().upper() for btn in actions_content.split(',') if btn.strip().upper() in valid_buttons]
        else:
            # Fallback: try to parse the entire response as comma-separated actions
            actions = [btn.strip().upper() for btn in action_response.split(',') if btn.strip().upper() in valid_buttons]
    except Exception as e:
        logger.warning(f"[ACTION] Error parsing actions: {e}")
        # Fallback: try to parse the entire response as comma-separated actions
        actions = [btn.strip().upper() for btn in action_response.split(',') if btn.strip().upper() in valid_buttons]
    
    print(f"Parsed actions: {actions}")
    if len(actions) == 0:
        print("❌ No valid actions parsed - using default 'B'")
    print("-" * 80 + "\n")
    
    # Limit to maximum 10 actions and prevent excessive repetition
    actions = actions[:10]
    
    # If no valid actions found, make intelligent default based on state
    if not actions:
        if game_data.get('in_battle', False):
            actions = ['B']  # Attack in battle
        elif party_health['total_count'] == 0:
            actions = ['A', 'A', 'A']  # Try to progress dialogue/menu
        else:
            actions = [random.choice(['B'])]  # Random exploration
    
    logger.info(f"[ACTION] Actions decided: {', '.join(actions)}")

    # return raw action_response for logging in wandb weave
    return actions, action_response 