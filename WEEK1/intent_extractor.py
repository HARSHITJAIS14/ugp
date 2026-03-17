import json
from pydantic import BaseModel
from typing import Optional, List, Dict


class DiscoveryIntent(BaseModel):
    mof_filters: Dict
    gas: Optional[str] = None
    elements: Optional[List[str]] = None

class SimulationIntent(BaseModel):
    simulation_type: Optional[str]
    system: Dict
    parameters: Dict
    component: Dict

class UnifiedIntent(BaseModel):
    discovery: DiscoveryIntent
    simulation: SimulationIntent

from raspa_logger import RaspaLogger
import time

class IntentExtractor:

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.logger = RaspaLogger()
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.model = None

    def _prompt(self):
        return """
        You are an expert Molecular Simulation Data Engineer.
        Your task is to extract simulation parameters for RASPA software from natural language and
        Extract MOF structural search filters from user input.

        
        RULES:
        1. Output ONLY valid JSON. No markdown formatting.
        2. Map missing values to null.
        3. Convert pressures to Pascals (e.g., 1 bar = 100000 Pa).
        4. If the user gives a range of pressures, output them as a list.
        5. Do NOT invent unsupported properties.
        6. Interpret qualitative language intelligently:
        - "high surface area" → sa_m2g_min = 1500
        - "high porosity" → vf_min = 0.6
        - "microporous" → pld_max = 2.0
        - "large pore" → lcd_min = 10.0
        and others as needed based on common MOF terminology.
        7. Default limit = 20 if not specified.
        
        TARGET JSON STRUCTURE:
        {
          "simulation": {
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
        },
        "discovery": {
        {
            "mofid": null,
            "mofkey": null,
            "name": null,
            "database": null,
            "vf_min": null,
            "vf_max": null,
            "lcd_min": null,
            "lcd_max": null,
            "pld_min": null,
            "pld_max": null,
            "sa_m2g_min": null,
            "sa_m2g_max": null,
            "sa_m2cm3_min": null,
            "sa_m2cm3_max": null,
            "pressure_unit": null,
            "loading_unit": null,
            "limit": 20
            }
            }
                    """
#         return """
# Extract both:

# 1. MOF discovery filters
# 2. RASPA simulation parameters

# Return JSON:

# {
#   "discovery": {
#     "mof_filters": { ... },
#     "gas": "...",
#     "elements": ["..."]
#   },
#   "simulation": {
#     "simulation_type": "...",
#     "system": { ... },
#     "parameters": { ... },
#     "component": { ... }
#   }
# }
# Return JSON only.
# """

    # def extract(self, user_input, mock=False):

    #     if mock or not self.model:
    #         return {
    #             "discovery": {
    #                 "mof_filters": {"vf_min": 0.6, "pld_max": 2.0, "limit": 20},
    #                 "gas": "CO2",
    #                 "elements": None
    #             },
    #             "simulation": {
    #                 "simulation_type": "MonteCarlo",
    #                 "system": {"temperature": 298},
    #                 "parameters": {},
    #                 "component": {"name": "CO2"}
    #             }
    #         }

    #     response = self.model.generate_content(
    #         f"{self._prompt()}\n\nUSER INPUT: {user_input}"
    #     )

    #     clean = response.text.replace("```json","").replace("```","").strip()
    #     return json.loads(clean)
    def extract(self, user_input: str, request_id: Optional[int] = None, mock: bool = False):

        # ------------------------------
        # MOCK MODE
        # ------------------------------
        if mock or not self.model:
            return {
                "simulation": {
                    "simulation_type": "MonteCarlo",
                    "system": {
                        "framework_name": None,
                        "temperature": 298.0,
                        "pressure_list": [100000],
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
                        "name": "CO2",
                        "mole_fraction": 1.0
                    }
                },
                "discovery": {
                    "mofid": None,
                    "mofkey": None,
                    "name": None,
                    "database": None,
                    "vf_min": 0.6,
                    "vf_max": None,
                    "lcd_min": None,
                    "lcd_max": None,
                    "pld_min": None,
                    "pld_max": 2.0,
                    "sa_m2g_min": 1500,
                    "sa_m2g_max": None,
                    "sa_m2cm3_min": None,
                    "sa_m2cm3_max": None,
                    "pressure_unit": None,
                    "loading_unit": None,
                    "limit": 20
                }
            }

        # ------------------------------
        # REAL LLM CALL
        # ------------------------------
        try:
            response = self.model.generate_content(
                f"{self._prompt()}\n\nUSER INPUT: {user_input}"
            )

            clean = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)

            # Optional safety: enforce keys exist
            if "simulation" not in data:
                raise ValueError("Missing 'simulation' key in LLM output")

            if "discovery" not in data:
                raise ValueError("Missing 'discovery' key in LLM output")

            return data

        except Exception as e:
            print("❌ Intent extraction error:", e)

            # Safe fallback
            return {
                "simulation": {
                    "simulation_type": "MonteCarlo",
                    "system": {
                        "framework_name": None,
                        "temperature": None,
                        "pressure_list": None,
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
                        "name": None,
                        "mole_fraction": 1.0
                    }
                },
                "discovery": {
                    "mofid": None,
                    "mofkey": None,
                    "name": None,
                    "database": None,
                    "vf_min": None,
                    "vf_max": None,
                    "lcd_min": None,
                    "lcd_max": None,
                    "pld_min": None,
                    "pld_max": None,
                    "sa_m2g_min": None,
                    "sa_m2g_max": None,
                    "sa_m2cm3_min": None,
                    "sa_m2cm3_max": None,
                    "pressure_unit": None,
                    "loading_unit": None,
                    "limit": 20
                }
            }

