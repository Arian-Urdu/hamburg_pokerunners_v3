"""
Agent modules for Pokemon Emerald speedrunning agent
"""

from pyparsing import deque
from utils.vlm import VLM
from .action import action_step
from .memory import memory_step
from .perception import perception_step
from .planning import planning_step
from .simple import SimpleAgent, get_simple_agent, simple_mode_processing_multiprocess, configure_simple_agent_defaults


class Agent:
    """
    Unified agent interface that encapsulates all agent logic.
    The client just calls agent.step(game_state) and gets back an action.
    """
    
    def __init__(self, args=None):
        """
        Initialize the agent based on configuration.
        
        Args:
            args: Command line arguments with agent configuration
        """
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"
        simple_mode = args.simple if args else False
        
        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name)
        print(f"   VLM: {backend}/{model_name}")
        print(self.vlm)
        
        # Initialize agent mode
        self.simple_mode = simple_mode
        if simple_mode:
            # Use global SimpleAgent instance to enable checkpoint persistence
            self.simple_agent = get_simple_agent(self.vlm)
            print(f"   Mode: Simple (direct frame->action)")
        else:
            # Four-module agent context
            self.context = {
                'perception_output': None,
                'planning_output': None,
                'memory': [],
                'action': [],
                'observation_buffer': deque(maxlen=10)  # Store last 10 observations
            }
            print(f"   Mode: Four-module architecture")
    
    def step(self, game_state):
        """
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - audio: Dict with audio observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and optionally 'reasoning'
        """
        if self.simple_mode:
            # Simple mode - delegate to SimpleAgent
            return self.simple_agent.step(game_state)
        else:
            # Four-module processing
            try:
                # 1. Perception - understand what's happening
                print('HERE BEFORE PERCEPTION STEP')
                print(self.vlm)
                perception_output= perception_step(
                    game_state.get('frame'),
                    game_state, 
                    self.vlm, 
                    
                )
                self.context['perception_output'] = perception_output

                # add perception to observation buffer
                self.context['observation_buffer'].append({
                    "frame_id": game_state.get('frame_id', -1),
                    "observation": perception_output,
                    "state": game_state
                })
                
                # 2. Planning - decide strategy
                planning_output = planning_step(
                    self.context,
                    self.context.get('planning_output', None),
                    game_state,
                    self.vlm
                )
                self.context['planning_output'] = planning_output
                
                # 3. Memory - update context
                memory_output = memory_step(
                    self.context.get('memory', []),
                    self.context.get('planning_output', None),
                    self.context.get('action', []),
                    list(self.context.get('observation_buffer', [])),
                )
                self.context['memory'] = memory_output
                
                # 4. Action - choose button press
                action_output, _ = action_step(
                    self.context.get('memory', []),
                    self.context.get('planning_output', None),
                    self.context.get('perception_output', None),
                    game_state,
                    self.context.get('action', []),
                    self.vlm
                )

                 # Store action result
                self.context['action'].append(action_output)
                
                # Keep recent actions reasonable size
                if len(self.context['action']) > 40:
                    self.context['action'] = self.context['action'][-40:]

                return action_output
                
            except Exception as e:
                print(f"❌ Agent error: {e}")
                return None


__all__ = [
    'Agent',
    'action_step',
    'memory_step', 
    'perception_step',
    'planning_step',
    'SimpleAgent',
    'get_simple_agent',
    'simple_mode_processing_multiprocess',
    'configure_simple_agent_defaults'
]