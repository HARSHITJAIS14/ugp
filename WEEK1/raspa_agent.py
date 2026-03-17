import json
import os
import time
import subprocess
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from raspa_logger import RaspaLogger
from mofdb_client import fetch

# --- 1. Pydantic Models for LLM Output Validation ---

class SystemParams(BaseModel):
    framework_name: Optional[str] = Field(None, description="Name of the MOF/Zeolite framework")
    temperature: Optional[float] = Field(None, description="External temperature in Kelvin")
    pressure_list: Optional[List[float]] = Field(None, description="List of pressures in Pascals")
    unit_cells: Optional[List[int]] = Field(None, description="Array of unit cells [x, y, z]")
    helium_void_fraction: Optional[float] = Field(None, description="Helium void fraction of the framework")

class SimulationParams(BaseModel):
    number_of_cycles: Optional[int] = Field(None, description="Total number of Monte Carlo cycles")
    initialization_cycles: Optional[int] = Field(None, description="Cycles dedicated to initialization")
    forcefield: Optional[str] = Field(None, description="Forcefield to use (e.g., UFF, Dreiding)")
    cutoff: Optional[float] = Field(None, description="Cutoff radius in Angstroms")

class ComponentParams(BaseModel):
    name: Optional[str] = Field(None, description="Name of the adsorbate molecule")
    mole_fraction: Optional[float] = Field(1.0, description="Mole fraction of the component")

class RaspaRequest(BaseModel):
    simulation_type: str = Field("MonteCarlo", description="Type of simulation")
    system: SystemParams
    parameters: SimulationParams
    component: ComponentParams

# --- 2. Default Configuration (The Logic Layer) ---

DEFAULT_CONFIG = {
    "SimulationType": "MonteCarlo",
    "NumberOfCycles": 25000,
    "NumberOfInitializationCycles": 5000,
    "PrintEvery": 1000,
    "RestartFile": "no",
    "Forcefield": "TraPPE",
    "CutOff": 12.0,
    "ChargeMethod": "Ewald",
    "EwaldPrecision": 1e-6,
    "Framework": {
        # "FrameworkName": "MFI_SI",  # Fallback
        "FrameworkName": "AQOLOJ_clean",  # For mofdb API Testing
        "UnitCells": [2, 2, 2],
        "HeliumVoidFraction": 0.29,
        "ExternalTemperature": 298.0,
        "ExternalPressure": [100000]  # 1 bar in Pascals  
    },
    "Component": {
        "Name": "CO2",
        "MoleculeDefinition": "TraPPE",
        "TranslationProbability": 0.5,
        "RotationProbability": 0.5,
        "ReinsertionProbability": 0.5,
        "SwapProbability": 1.0,
        "CreateNumberOfMolecules": 0
    }
}

# --- 3. The Agent Class ---

class RaspaAgent:
    def __init__(self, 
                #  api_key: str = None
                 ):
        # self.api_key = api_key
        self.logger = RaspaLogger()
        # If using Google Gemini:
        # if self.api_key:
        #     import google.generativeai as genai
        #     genai.configure(api_key=self.api_key)
        #     self.model = genai.GenerativeModel('gemini-2.5-flash') # Or pro

    # def _get_system_prompt(self) -> str:
    #     return """
    #     You are an expert Molecular Simulation Data Engineer.
    #     Your task is to extract simulation parameters for RASPA software from natural language.
        
    #     RULES:
    #     1. Output ONLY valid JSON. No markdown formatting.
    #     2. Map missing values to null.
    #     3. Convert pressures to Pascals (e.g., 1 bar = 100000 Pa).
    #     4. If the user gives a range of pressures, output them as a list.
        
    #     TARGET JSON STRUCTURE:
    #     {
    #       "simulation_type": "MonteCarlo",
    #       "system": {
    #         "framework_name": "String or null",
    #         "temperature": "Float or null",
    #         "pressure_list": "Array of Floats or null",
    #         "unit_cells": "Array [x, y, z] or null",
    #         "helium_void_fraction": "Float or null"
    #       },
    #       "parameters": {
    #         "number_of_cycles": "Integer or null",
    #         "initialization_cycles": "Integer or null",
    #         "forcefield": "String or null",
    #         "cutoff": "Float or null"
    #       },
    #       "component": {
    #         "name": "String (e.g., 'CO2', 'methane') or null",
    #         "mole_fraction": "Float"
    #       }
    #     }
    #     """

    # def parse_with_llm(self, user_input: str, mock: bool = False, request_id: int = None) -> RaspaRequest:
    #     """
    #     Extracts parameters using LLM or returns a mock response for testing.
    #     """
    #     step_start = time.time()
        
    #     if mock or not self.api_key:
    #         # Mock behavior for testing without API key
    #         print(">> [MOCK MODE] Simulating LLM extraction...")
    #         mock_data = {
    #             "simulation_type": "MonteCarlo",
    #             "system": {
    #                 "framework_name": "HKUST-1" if "HKUST" in user_input else None,
    #                 "temperature": 300.0 if "300" in user_input else None,
    #                 "pressure_list": [10000, 100000] if "range" in user_input else None,
    #                 "unit_cells": None,
    #                 "helium_void_fraction": None
    #             },
    #             "parameters": {
    #                 "number_of_cycles": None,
    #                 "initialization_cycles": None,
    #                 "forcefield": None,
    #                 "cutoff": None
    #             },
    #             "component": {
    #                 "name": "methane" if "methane" in user_input.lower() else None,
    #                 "mole_fraction": 1.0
    #             }
    #         }
    #         result = RaspaRequest(**mock_data)
    #         if request_id:
    #             self.logger.log_step(request_id, "llm_called_mock", {"input": user_input, "result": result.dict()}, time.time() - step_start)
    #         return result

    #     # Real LLM Call
    #     try:
    #         full_prompt = f"{self._get_system_prompt()}\n\nUSER INPUT: {user_input}"
    #         response = self.model.generate_content(full_prompt)
    #         # clean response text (remove ```json if present)
    #         clean_text = response.text.replace("```json", "").replace("```", "").strip()
    #         data = json.loads(clean_text)
    #         result = RaspaRequest(**data)
    #         if request_id:
    #             self.logger.log_step(request_id, "llm_called", {"prompt": full_prompt[:100] + "...", "result": result.dict()}, time.time() - step_start)
    #         return result
    #     except Exception as e:
    #         print(f"Error calling LLM: {e}")
    #         if request_id:
    #             self.logger.log_step(request_id, "llm_error", {"error": str(e)}, time.time() - step_start)
    #         return RaspaRequest(
    #             simulation_type="MonteCarlo", 
    #             system=SystemParams(), 
    #             parameters=SimulationParams(), 
    #             component=ComponentParams()
    #         )

    # ---------------------------------
    # Merge Defaults
    # ---------------------------------
    def merge_defaults(self, simulation_block: Dict[str, Any], selected_framework: str):

        config = DEFAULT_CONFIG.copy()
        config["Framework"] = DEFAULT_CONFIG["Framework"].copy()
        config["Component"] = DEFAULT_CONFIG["Component"].copy()

        # Inject framework
        config["Framework"]["FrameworkName"] = selected_framework

        # ---- System ----
        system = simulation_block.get("system", {})
        parameters = simulation_block.get("parameters", {})
        component = simulation_block.get("component", {})

        if system.get("temperature") is not None:
            config["Framework"]["ExternalTemperature"] = system["temperature"]

        if system.get("pressure_list") is not None:
            config["Framework"]["ExternalPressure"] = system["pressure_list"]

        if system.get("unit_cells") is not None:
            config["Framework"]["UnitCells"] = system["unit_cells"]

        if system.get("helium_void_fraction") is not None:
            config["Framework"]["HeliumVoidFraction"] = system["helium_void_fraction"]

        # ---- Parameters ----
        if parameters.get("number_of_cycles") is not None:
            config["NumberOfCycles"] = parameters["number_of_cycles"]

        if parameters.get("initialization_cycles") is not None:
            config["NumberOfInitializationCycles"] = parameters["initialization_cycles"]

        if parameters.get("forcefield") is not None:
            config["Forcefield"] = parameters["forcefield"]

        if parameters.get("cutoff") is not None:
            config["CutOff"] = parameters["cutoff"]

        # ---- Component ----
        if component.get("name") is not None:
            config["Component"]["Name"] = component["name"]

            if component["name"].lower() in ["co2", "methane"]:
                config["Component"]["MoleculeDefinition"] = "TraPPE"

        return config

    # ---------------------------------
    # Fetch CIF
    # ---------------------------------
    def fetch_and_save_cif(self, framework_name: str):

        print(f"🔍 Fetching CIF for {framework_name}")

        gen = fetch(name=framework_name, limit=1)
        mof = next(gen, None)

        if mof is None:
            print("❌ CIF not found.")
            return None

        # os.makedirs("framework", exist_ok=True)
        # filename = os.path.join("framework", f"{mof.name}.cif")
        raspa_cif_dir = os.path.join(os.environ["RASPA_DIR"],"share", "raspa", "structures", "cif")
        os.makedirs(raspa_cif_dir, exist_ok=True)
        filename = os.path.join(raspa_cif_dir, f"{mof.name}.cif") # Save directly to RASPA's cif directory

        with open(filename, "w") as f:
            f.write(mof.cif)

        print(f"✔ CIF saved: {filename}")
        return filename

    def validate_forcefield(self, forcefield_name):
        ff_path = os.path.join(os.environ.get("RASPA_DIR", ""), 
                            "share/raspa/forcefield", 
                            forcefield_name)
        return os.path.isdir(ff_path)

    # ---------------------------------
    # Generate RASPA Input File
    # ---------------------------------
    def generate_raspa_file_content(self, config: Dict[str, Any]) -> str:

        pressure_list = config["Framework"]["ExternalPressure"]
        pressure_str = " ".join(str(p) for p in pressure_list)

        uc = config["Framework"]["UnitCells"]
        uc_str = f"{uc[0]} {uc[1]} {uc[2]}"

        return f"""
SimulationType                {config['SimulationType']}
NumberOfCycles                {config['NumberOfCycles']}
NumberOfInitializationCycles  {config['NumberOfInitializationCycles']}
PrintEvery                    {config['PrintEvery']}
RestartFile                   {config['RestartFile']}

Forcefield                    {config['Forcefield']}
CutOff                        {config['CutOff']}
ChargeMethod                  {config['ChargeMethod']}
EwaldPrecision                {config['EwaldPrecision']}
RemoveAtomNumberCodeFromLabel yes

Framework 0
FrameworkName {config['Framework']['FrameworkName']}
UnitCells {uc_str}
HeliumVoidFraction {config['Framework']['HeliumVoidFraction']}
ExternalTemperature {config['Framework']['ExternalTemperature']}
ExternalPressure {pressure_str}

Component 0 MoleculeName              {config['Component']['Name']}
            MoleculeDefinition        {config['Component']['MoleculeDefinition']}
            TranslationProbability    {config['Component']['TranslationProbability']}
            RotationProbability       {config['Component']['RotationProbability']}
            ReinsertionProbability    {config['Component']['ReinsertionProbability']}
            SwapProbability           {config['Component']['SwapProbability']}
            CreateNumberOfMolecules   {config['Component']['CreateNumberOfMolecules']}
"""

    # ---------------------------------
    # Run Full Simulation
    # ---------------------------------
    def run(self, simulation_block: Dict[str, Any], selected_framework: str, request_id: Optional[int] = None):
        import uuid
        import shutil
        from plot_isotherm import generate_isotherm_plot

        print(f"\n🚀 Preparing simulation for {selected_framework}")

        config = self.merge_defaults(simulation_block, selected_framework)

        cif_file = self.fetch_and_save_cif(selected_framework)
        if not cif_file:
            print("Simulation aborted.")
            return

        forcefield = config["Forcefield"]
        if not self.validate_forcefield(forcefield):
            print(f"❌ Forcefield '{forcefield}' not found.")
            print("Available forcefields:")
            ff_dir = os.path.join(os.environ["RASPA_DIR"], "share/raspa/forcefield")
            for ff in os.listdir(ff_dir):
                print(" -", ff)
            chosen = input("Enter forcefield to use: ").strip()
            config["Forcefield"] = chosen

        # Handle multiple pressures for an isotherm
        pressures = config["Framework"]["ExternalPressure"]
        if not isinstance(pressures, list):
            pressures = [pressures]

        # For simplicity, if len == 1, it runs a single point. If len > 1, it's an isotherm run.
        # But we also allow a manual override logic if the user passes `is_isotherm=True`
        is_isotherm = len(pressures) > 1
        
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = os.path.join("Output", f"job_{job_id}")
        
        if is_isotherm:
            print(f"\n📈 Isotherm Multi-Run Detected: {len(pressures)} points.")
            print(f"📁 Job Directory: {job_dir}")
            os.makedirs(job_dir, exist_ok=True)
            
        for i, p in enumerate(pressures):
            if is_isotherm:
                print(f"\n--- Running Point {i+1}/{len(pressures)} : Pressure = {p} Pa ---")
            
            # Temporarily set this config specific pressure
            config["Framework"]["ExternalPressure"] = [p]
            
            content = self.generate_raspa_file_content(config)

            with open("simulation.input", "w") as f:
                f.write(content)

            if not is_isotherm:
                print("\n📄 Generated RASPA input file (Preview):")
                print("-" * 40)
                print(content[:500])
                print("-" * 40)
                print("\n🚀 Launching RASPA simulation...")
                print("   This may take a few minutes depending on system size.")
            
            # Run RASPA
            process = subprocess.Popen(
                ["simulate", "simulation.input"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Only print every line for single runs to avoid massive spam
            for line in process.stdout:
                if not is_isotherm:
                    print(line.strip())

            process.wait()

            if process.returncode != 0:
                print(f"❌ Simulation failed at pressure {p} Pa.")
                if is_isotherm:
                     print("Aborting remaining points.")
                     break
            else:
                if not is_isotherm:
                    print("✅ Simulation completed successfully.")
                else:
                    print(f"✅ Point {i+1} completed.")
                    
                # Move outputs to isolated job directory if isotherm
                if is_isotherm:
                    system_0_dir = os.path.join("Output", "System_0")
                    if os.path.exists(system_0_dir):
                        for file in os.listdir(system_0_dir):
                            if file.endswith('.data'):
                                # rename to avoid clashes
                                src = os.path.join(system_0_dir, file)
                                dst = os.path.join(job_dir, f"p{p}_{file}")
                                shutil.move(src, dst)

        # Generate plot if isotherm
        if is_isotherm and process.returncode == 0:
            print(f"\n📊 Combining {len(pressures)} outputs to generate Isotherm Plot...")
            generate_isotherm_plot(job_dir, "Plots")



    # def merge_defaults(self, request: RaspaRequest, request_id: int = None) -> Dict[str, Any]:
    #     """
    #     Merges the LLM-extracted data with the DEFAULT_CONFIG.
    #     """
    #     step_start = time.time()
    #     config = DEFAULT_CONFIG.copy()
        
    #     # Deep copy structure for framework and component to avoid mutating defaults
    #     config["Framework"] = DEFAULT_CONFIG["Framework"].copy()
    #     config["Component"] = DEFAULT_CONFIG["Component"].copy()

    #     # 1. Merge Global Parameters
    #     if request.parameters.number_of_cycles:
    #         config["NumberOfCycles"] = request.parameters.number_of_cycles
    #     if request.parameters.initialization_cycles:
    #         config["NumberOfInitializationCycles"] = request.parameters.initialization_cycles
    #     if request.parameters.forcefield:
    #         config["Forcefield"] = request.parameters.forcefield
    #     if request.parameters.cutoff:
    #         config["CutOff"] = request.parameters.cutoff

    #     # 2. Merge Framework System
    #     if request.system.framework_name:
    #         config["Framework"]["FrameworkName"] = request.system.framework_name
    #     if request.system.unit_cells:
    #         config["Framework"]["UnitCells"] = request.system.unit_cells
    #     if request.system.helium_void_fraction:
    #         config["Framework"]["HeliumVoidFraction"] = request.system.helium_void_fraction
    #     if request.system.temperature:
    #         config["Framework"]["ExternalTemperature"] = request.system.temperature
    #     if request.system.pressure_list:
    #         config["Framework"]["ExternalPressure"] = request.system.pressure_list

    #     # 3. Merge Component
    #     if request.component.name:
    #         config["Component"]["Name"] = request.component.name
    #         # Simple heuristic for definition name
    #         if request.component.name.lower() == "co2":
    #             config["Component"]["MoleculeDefinition"] = "TraPPE"
    #         elif request.component.name.lower() == "methane":
    #             config["Component"]["MoleculeDefinition"] = "TraPPE"
    #         else:
    #             config["Component"]["MoleculeDefinition"] = "ExampleDefinitions"

    #     if request_id:
    #         self.logger.log_step(request_id, "defaults_merged", {"config": config}, time.time() - step_start)

    #     return config
    
    # def fetch_and_save_cif(self, framework_name: str, request_id: int = None):
    #     """
    #     Fetch CIF from mofdb and save locally.
    #     """
    #     if not framework_name:
    #         return None

    #     print(f"🔍 Fetching CIF for framework: {framework_name}")

    #     try:
    #         gen = fetch(name=framework_name, limit=1)

    #         mof = next(gen, None)   # SAFE: does not raise StopIteration

    #         if mof is None:
    #             print("⚠ No matching MOF found in database.")
    #             return None

    #         os.makedirs("framework", exist_ok=True)
    #         filename = os.path.join("framework", f"{mof.name}.cif")

    #         with open(filename, "w") as f:
    #             f.write(mof.cif)

    #         print(f"✔ CIF saved as {filename}")

    #         if request_id:
    #             self.logger.log_step(
    #                 request_id,
    #                 "cif_downloaded",
    #                 {"framework": mof.name, "file": filename},
    #                 0
    #             )

    #         return filename

        # except Exception as e:
        #     print(f"❌ Error fetching CIF: {e}")

        #     if request_id:
        #         self.logger.log_step(
        #             request_id,
        #             "cif_download_error",
        #             {"error": str(e)},
        #             0
        #         )

        #     return 
    
    # def safe_fetch(self,**kwargs):
    #     try:
    #         gen = fetch(**kwargs, telemetry=False)
    #         while True:
    #             try:
    #                 yield next(gen)
    #             except StopIteration:
    #                 break
    #     except RuntimeError:
    #         # Handles the "generator raised StopIteration"
    #         return

    # def resolve_framework_interactively(self, framework_name: str):
    #     """
    #     Suggest similar frameworks or allow manual input if none found.
    #     """
    #     print(f"❌ Framework '{framework_name}' not found in mofdb.")

    #     # Try to find similar candidates
    #     candidates = []
    #     try:
    #         prefix = framework_name.split('-')[0]
    #         for m in self.safe_fetch(name=prefix, limit=10):
    #             candidates.append(m.name)
    #     except:
    #         pass

    #     if candidates:
    #         print("\n🔎 Did you mean one of these?")
    #         for i, c in enumerate(candidates, 1):
    #             print(f"  {i}. {c}")
    #     else:
    #         print("\n⚠ No similar frameworks found in mofdb.")

    #     # Ask user to choose or enter new one
    #     print("\n👉 Enter the correct framework name (or type 'exit' to abort):")
    #     user_choice = input("> ").strip()

    #     if user_choice.lower() == "exit":
    #         return None

    #     return user_choice



#     def generate_raspa_file_content(self, config: Dict[str, Any], request_id: int = None) -> str:
#         """
#         Generates the .txt content in RASPA format.
#         """
#         step_start = time.time()
#         # Format lists for RASPA (space separated)
#         p_list = config['Framework']['ExternalPressure']
#         if isinstance(p_list, list):
#             pressure_str = " ".join(str(p) for p in p_list)
#         else:
#             pressure_str = str(p_list)

#         uc = config['Framework']['UnitCells']
#         uc_str = f"{uc[0]} {uc[1]} {uc[2]}"

#         # Construct file content
#         content = f"""SimulationType                {config['SimulationType']}
# NumberOfCycles                {config['NumberOfCycles']}
# NumberOfInitializationCycles  {config['NumberOfInitializationCycles']}
# PrintEvery                    {config['PrintEvery']}
# RestartFile                   {config['RestartFile']}

# Forcefield                    {config['Forcefield']}
# CutOff                        {config['CutOff']}
# ChargeMethod                  {config['ChargeMethod']}
# EwaldPrecision                {config['EwaldPrecision']}
# RemoveAtomNumberCodeFromLabel yes

# Framework 0
# FrameworkName {config['Framework']['FrameworkName']}
# UnitCells {uc_str}
# HeliumVoidFraction {config['Framework']['HeliumVoidFraction']}
# ExternalTemperature {config['Framework']['ExternalTemperature']}
# ExternalPressure {pressure_str}

# ComputeNumberOfMoleculesHistogram yes
# WriteNumberOfMoleculesHistogramEvery 5000
# NumberOfMoleculesHistogramSize 1000
# NumberOfMoleculesRange 100

# ComputeEnergyHistogram yes
# WriteEnergyHistogramEvery 5000
# EnergyHistogramSize 400
# EnergyHistogramLowerLimit -50000
# EnergyHistogramUpperLimit 5000

# Component 0 MoleculeName              {config['Component']['Name']}
#             MoleculeDefinition        {config['Component']['MoleculeDefinition']}
#             TranslationProbability    {config['Component']['TranslationProbability']}
#             RotationProbability       {config['Component']['RotationProbability']}
#             ReinsertionProbability    {config['Component']['ReinsertionProbability']}
#             SwapProbability           {config['Component']['SwapProbability']}
#             CreateNumberOfMolecules   {config['Component']['CreateNumberOfMolecules']}
# """
#         if request_id:
#             self.logger.log_step(request_id, "raspa_file_generated", {"content_length": len(content)}, time.time() - step_start)

#         return content

    # def run(self, user_input: str, mock: bool = False, output_filename: str = None):
    #     """
    #     Orchestrates the whole process with logging.
    #     """
    #     print(f"--- Processing Request: '{user_input}' ---")
        
    #     # Log the request
    #     request_id = self.logger.log_request({"query": user_input})
    #     print(f"[Request ID: {request_id}]")
        
    #     # 1. LLM Extraction
    #     step_start = time.time()
    #     extracted_obj = self.parse_with_llm(user_input, mock=mock, request_id=request_id)
    #     print(f"✔ Extracted Entities: {extracted_obj.dict(exclude_none=True)}")
    #     self.logger.log_step(request_id, "extraction_complete", {"extracted": extracted_obj.dict(exclude_none=True)}, time.time() - step_start)
        
    #     # 2. Defaults Merging
    #     final_config = self.merge_defaults(extracted_obj, request_id=request_id)
    #     print("✔ Applied Defaults.")

    #     # 2.5 Fetch CIF file
    #     framework_name = final_config["Framework"]["FrameworkName"]
    #     cif_file = self.fetch_and_save_cif(framework_name, request_id=request_id)
    #     while cif_file is None:
    #         new_name = self.resolve_framework_interactively(framework_name)

    #         if new_name is None:
    #             print("🛑 Simulation aborted.")
    #             return

    #         framework_name = new_name
    #         cif_file = self.fetch_and_save_cif(framework_name, request_id=request_id)

    #     final_config["Framework"]["FrameworkName"] = framework_name
    #     print("✔ CIF Fetched.")
        
    #     # 3. File Generation
    #     step_start = time.time()
    #     content = self.generate_raspa_file_content(final_config, request_id=request_id)
    #     print(f"✔ RASPA file generated ({len(content)} chars)")
        
    #     # 4. File Writing
    #     if output_filename:
    #         step_start = time.time()
    #         with open(output_filename, "w") as f:
    #             f.write(content)
    #         file_write_time = time.time() - step_start
    #         print(f"✔ File saved to: {output_filename}")
    #         self.logger.log_step(request_id, "file_written", {"filename": output_filename, "size": len(content)}, file_write_time)
        
    #     # Log final result
    #     self.logger.log_final_result(request_id, {"output_file": output_filename, "content_size": len(content)})
        
    #     return content