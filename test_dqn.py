import os
import sys
import shutil
from src.config import SimulationConfig
from src.core.spgg_model import SPGG
from src.core.state_strategies import ReputationStateProvider

def test_dqn_integration():
    print("Testing DQN Integration...")
    
    # Enable DQN
    config = SimulationConfig(
        L=20, 
        iterations=50, 
        use_dqn=True,
        dqn_update_freq=5,
        dqn_batch_size=32
    )
    
    state_provider = ReputationStateProvider(config)
    
    folder = "test_dqn_results"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    os.makedirs(os.path.join(folder, "data"))
    os.makedirs(os.path.join(folder, "plots", "snapshots"))
    
    spgg = SPGG(config, state_provider, folder=folder)
    
    # Run simulation
    try:
        spgg.run(os.path.join(folder, "data", "experiment_data.h5"))
        print("SUCCESS: Simulation with DQN completed without errors.")
    except Exception as e:
        print(f"FAILURE: Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dqn_integration()
