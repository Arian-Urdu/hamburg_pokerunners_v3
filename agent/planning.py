import logging
import weave
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary
from agent.system_prompt import system_prompt

# Set up module logging
logger = logging.getLogger(__name__)

@weave.op()
def planning_step(context, current_plan, state_data, vlm):
    """
    Decide and update your high-level plan based on memory context, current state, and the need for slow thinking.
    Returns updated plan.
    """
    # Get formatted state context
    state_context = format_state_for_llm(state_data)
    state_summary = format_state_summary(state_data)
    
    logger.info("[PLANNING] Starting planning step")
    logger.info(f"[PLANNING] State: {state_summary}")
    logger.info(f"[PLANNING] Slow thinking needed: not doing slow thinking anymore")
    
    # Check if current plan is accomplished
    if current_plan:
        plan_check_prompt = f"""PLAN ASSESSMENT TASK
You are the agent playing Pokemon Emerald. Assess your current situation and plan progress.
        
You will be provided with the following information:
<context>
    <game_state>
    {state_context}
    </game_state>

    <current_plan>
    {current_plan}
    </current_plan>

    <memory_context>
    {context.get('memory', [])}
    </memory_context>

    <current_observation>
    {context.get('perception_output', [])}
    </current_observation>
</context>

Considering your current location, pokemon party, money, traversability, and recent actions:
        
1. OBJECTIVES:
- Specific steps to be achieved through the immediate goal
- Account for your current pokemon party health and levels
- Consider terrain: avoid/seek tall grass, navigate around obstacles
        
2. NEXT GOAL (next few actions): What should you focus on right now? Consider:
- If in battle: What's your battle strategy based on pokemon HP/levels?
- If on map: Navigate efficiently using traversability data
- If in menu/dialogue: How to progress efficiently (Press A)?
- Do you need to heal pokemon at Pokemon Center?
- Are there terrain obstacles (water, blocked paths) to navigate?

Then describe in detail the next objective and next goal as you are trying to speedrun prioritize making progress in-game and reaching the next area."""
        
        plan_status = vlm.get_text_query(system_prompt + plan_check_prompt, "PLANNING-ASSESSMENT")
        if "yes" in plan_status.lower():
            current_plan = None
            logger.info("[PLANNING] Current plan marked as completed")
    
    # Generate new plan if needed
    if current_plan is None:
        planning_prompt =  f"""PLAN CREATION TASK
You are the agent playing Pokemon Emerald. Assess your current situation and make an initial plan for the next goal.
        
You will be provided with the following information:
<context>
    <game_state>
    {state_context}
    </game_state>

    <memory_context>
    {context.get('memory', [])}
    </memory_context>

    <current_observation>
    {context.get('perception_output', [])}
    </current_observation>
</context>

Considering your current location, pokemon party, money, traversability, and recent actions:
        
1. OBJECTIVES:
- Specific steps to be achieved through the immediate goal
- Account for your current pokemon party health and levels
- Consider terrain: avoid/seek tall grass, navigate around obstacles
        
2. IMMEDIATE NEXT GOAL (next few actions): What should you focus on right now? Consider:
- If in battle: What's your battle strategy based on pokemon HP/levels?
- If on map: Navigate efficiently using traversability data
- If in menu/dialogue: How to progress efficiently (Press A)?
- Do you need to heal pokemon at Pokemon Center?
- Are there terrain obstacles (water, blocked paths) to navigate?
        
Then describe in detail the next immediate goal."""
        

        current_plan = vlm.get_text_query(system_prompt + planning_prompt, "PLANNING-CREATION")
        logger.info("[PLANNING] New plan created")
    
    logger.info(f"[PLANNING] Final plan: {current_plan[:300]}..." if len(current_plan) > 300 else f"[PLANNING] Final plan: {current_plan}")
    return current_plan 