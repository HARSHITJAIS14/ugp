import json
import os
import time
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from raspa_logger import RaspaLogger

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
    "Forcefield": "UFF",
    "CutOff": 12.0,
    "ChargeMethod": "Ewald",
    "EwaldPrecision": 1e-6,
    "Framework": {
        "FrameworkName": "MFI_SI",  # Fallback
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
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.logger = RaspaLogger()
        # If using Google Gemini:
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash') # Or pro

    def _get_system_prompt(self) -> str:
        return """
        You are an expert Molecular Simulation Data Engineer.
        Your task is to extract simulation parameters for RASPA software from natural language.
        
        RULES:
        1. Output ONLY valid JSON. No markdown formatting.
        2. Map missing values to null.
        3. Convert pressures to Pascals (e.g., 1 bar = 100000 Pa).
        4. If the user gives a range of pressures, output them as a list.
        
        TARGET JSON STRUCTURE:
        {
          "simulation_type": "MonteCarlo",
          "system": {
            "framework_name": "String or null",
            "temperature": "Float or null",
            "pressure_list": "Array of Floats or null",
            "unit_cells": "Array [x, y, z] or null",
            "helium_void_fraction": "Float or null"
          },
          "parameters": {
            "number_of_cycles": "Integer or null",
            "initialization_cycles": "Integer or null",
            "forcefield": "String or null",
            "cutoff": "Float or null"
          },
          "component": {
            "name": "String (e.g., 'CO2', 'methane') or null",
            "mole_fraction": "Float"
          }
        }
        """

    def parse_with_llm(self, user_input: str, mock: bool = False, request_id: int = None) -> RaspaRequest:
        """
        Extracts parameters using LLM or returns a mock response for testing.
        """
        step_start = time.time()
        
        if mock or not self.api_key:
            # Mock behavior for testing without API key
            print(">> [MOCK MODE] Simulating LLM extraction...")
            mock_data = {
                "simulation_type": "MonteCarlo",
                "system": {
                    "framework_name": "HKUST-1" if "HKUST" in user_input else None,
                    "temperature": 300.0 if "300" in user_input else None,
                    "pressure_list": [10000, 100000] if "range" in user_input else None,
                    "unit_cells": None,
                    "helium_void_fraction": None
                },
                "parameters": {
                    "number_of_cycles": None,
                    "initialization_cycles": None,
                    "forcefield": None,
                    "cutoff": None
                },
                "component": {
                    "name": "methane" if "methane" in user_input.lower() else None,
                    "mole_fraction": 1.0
                }
            }
            result = RaspaRequest(**mock_data)
            if request_id:
                self.logger.log_step(request_id, "llm_called_mock", {"input": user_input, "result": result.dict()}, time.time() - step_start)
            return result

        # Real LLM Call
        try:
            full_prompt = f"{self._get_system_prompt()}\n\nUSER INPUT: {user_input}"
            response = self.model.generate_content(full_prompt)
            # clean response text (remove ```json if present)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            result = RaspaRequest(**data)
            if request_id:
                self.logger.log_step(request_id, "llm_called", {"prompt": full_prompt[:100] + "...", "result": result.dict()}, time.time() - step_start)
            return result
        except Exception as e:
            print(f"Error calling LLM: {e}")
            if request_id:
                self.logger.log_step(request_id, "llm_error", {"error": str(e)}, time.time() - step_start)
            return RaspaRequest(
                simulation_type="MonteCarlo", 
                system=SystemParams(), 
                parameters=SimulationParams(), 
                component=ComponentParams()
            )

    def merge_defaults(self, request: RaspaRequest, request_id: int = None) -> Dict[str, Any]:
        """
        Merges the LLM-extracted data with the DEFAULT_CONFIG.
        """
        step_start = time.time()
        config = DEFAULT_CONFIG.copy()
        
        # Deep copy structure for framework and component to avoid mutating defaults
        config["Framework"] = DEFAULT_CONFIG["Framework"].copy()
        config["Component"] = DEFAULT_CONFIG["Component"].copy()

        # 1. Merge Global Parameters
        if request.parameters.number_of_cycles:
            config["NumberOfCycles"] = request.parameters.number_of_cycles
        if request.parameters.initialization_cycles:
            config["NumberOfInitializationCycles"] = request.parameters.initialization_cycles
        if request.parameters.forcefield:
            config["Forcefield"] = request.parameters.forcefield
        if request.parameters.cutoff:
            config["CutOff"] = request.parameters.cutoff

        # 2. Merge Framework System
        if request.system.framework_name:
            config["Framework"]["FrameworkName"] = request.system.framework_name
        if request.system.unit_cells:
            config["Framework"]["UnitCells"] = request.system.unit_cells
        if request.system.helium_void_fraction:
            config["Framework"]["HeliumVoidFraction"] = request.system.helium_void_fraction
        if request.system.temperature:
            config["Framework"]["ExternalTemperature"] = request.system.temperature
        if request.system.pressure_list:
            config["Framework"]["ExternalPressure"] = request.system.pressure_list

        # 3. Merge Component
        if request.component.name:
            config["Component"]["Name"] = request.component.name
            # Simple heuristic for definition name
            if request.component.name.lower() == "co2":
                config["Component"]["MoleculeDefinition"] = "TraPPE"
            elif request.component.name.lower() == "methane":
                config["Component"]["MoleculeDefinition"] = "TraPPE"
            else:
                config["Component"]["MoleculeDefinition"] = "ExampleDefinitions"

        if request_id:
            self.logger.log_step(request_id, "defaults_merged", {"config": config}, time.time() - step_start)

        return config

    def generate_raspa_file_content(self, config: Dict[str, Any], request_id: int = None) -> str:
        """
        Generates the .txt content in RASPA format.
        """
        step_start = time.time()
        # Format lists for RASPA (space separated)
        p_list = config['Framework']['ExternalPressure']
        if isinstance(p_list, list):
            pressure_str = " ".join(str(p) for p in p_list)
        else:
            pressure_str = str(p_list)

        uc = config['Framework']['UnitCells']
        uc_str = f"{uc[0]} {uc[1]} {uc[2]}"

        # Construct file content
        content = f"""SimulationType                {config['SimulationType']}
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

ComputeNumberOfMoleculesHistogram yes
WriteNumberOfMoleculesHistogramEvery 5000
NumberOfMoleculesHistogramSize 1000
NumberOfMoleculesRange 100

ComputeEnergyHistogram yes
WriteEnergyHistogramEvery 5000
EnergyHistogramSize 400
EnergyHistogramLowerLimit -50000
EnergyHistogramUpperLimit 5000

Component 0 MoleculeName              {config['Component']['Name']}
            MoleculeDefinition        {config['Component']['MoleculeDefinition']}
            TranslationProbability    {config['Component']['TranslationProbability']}
            RotationProbability       {config['Component']['RotationProbability']}
            ReinsertionProbability    {config['Component']['ReinsertionProbability']}
            SwapProbability           {config['Component']['SwapProbability']}
            CreateNumberOfMolecules   {config['Component']['CreateNumberOfMolecules']}
"""
        if request_id:
            self.logger.log_step(request_id, "raspa_file_generated", {"content_length": len(content)}, time.time() - step_start)

        return content

    def run(self, user_input: str, mock: bool = False, output_filename: str = None):
        """
        Orchestrates the whole process with logging.
        """
        print(f"--- Processing Request: '{user_input}' ---")
        
        # Log the request
        request_id = self.logger.log_request({"query": user_input})
        print(f"[Request ID: {request_id}]")
        
        # 1. LLM Extraction
        step_start = time.time()
        extracted_obj = self.parse_with_llm(user_input, mock=mock, request_id=request_id)
        print(f"✔ Extracted Entities: {extracted_obj.dict(exclude_none=True)}")
        self.logger.log_step(request_id, "extraction_complete", {"extracted": extracted_obj.dict(exclude_none=True)}, time.time() - step_start)
        
        # 2. Defaults Merging
        final_config = self.merge_defaults(extracted_obj, request_id=request_id)
        print("✔ Applied Defaults.")
        
        # 3. File Generation
        step_start = time.time()
        content = self.generate_raspa_file_content(final_config, request_id=request_id)
        print(f"✔ RASPA file generated ({len(content)} chars)")
        
        # 4. File Writing
        if output_filename:
            step_start = time.time()
            with open(output_filename, "w") as f:
                f.write(content)
            file_write_time = time.time() - step_start
            print(f"✔ File saved to: {output_filename}")
            self.logger.log_step(request_id, "file_written", {"filename": output_filename, "size": len(content)}, file_write_time)
        
        # Log final result
        self.logger.log_final_result(request_id, {"output_file": output_filename, "content_size": len(content)})
        
        return content