from raspa_agent import RaspaAgent
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    if "RASPA_DIR" in os.environ:
        os.environ["PATH"] = os.path.join(os.environ["RASPA_DIR"], "bin") + os.pathsep + os.environ.get("PATH", "")
    agent = RaspaAgent()
    
    # Mocking the JSON block normally provided by the LLM
    # We explicitly provide the 9 point pressure list the user asked for 
    simulation_block = {
        "simulation_type": "MonteCarlo",
        "system": {
            "temperature": 298.0,
            "pressure_list": [1000, 5000, 10000, 15000, 20000, 40000, 60000, 80000, 100000], 
        },
        "parameters": {
            "initialization_cycles": 500,
            "number_of_cycles": 1000,
            "forcefield": "ExampleMOFsForceField"
        },
        "component": {
            "name": "CO2",
            "mole_fraction": 1.0
        }
    }
    
    print("Running 9 point Isotherm Sweep on hMOF-6")
    agent.run(simulation_block, "hMOF-6")
