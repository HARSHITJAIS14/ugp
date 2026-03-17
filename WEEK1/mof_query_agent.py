import json
import time
import requests
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from mofdb_client import fetch
from raspa_logger import RaspaLogger


# ------------------------------
# 1️⃣ Pydantic Model for fetch()
# ------------------------------

class MofdbQuery(BaseModel):
    mofid: Optional[str] = None
    mofkey: Optional[str] = None
    name: Optional[str] = None
    database: Optional[str] = None

    vf_min: Optional[float] = None
    vf_max: Optional[float] = None

    lcd_min: Optional[float] = None
    lcd_max: Optional[float] = None

    pld_min: Optional[float] = None
    pld_max: Optional[float] = None

    sa_m2g_min: Optional[float] = None
    sa_m2g_max: Optional[float] = None

    sa_m2cm3_min: Optional[float] = None
    sa_m2cm3_max: Optional[float] = None

    pressure_unit: Optional[str] = None
    loading_unit: Optional[str] = None

    limit: Optional[int] = 20


# ------------------------------
# 2️⃣ Combined Discovery Agent
# ------------------------------

class MofDiscoveryAgent:

    def __init__(
        self,
        # api_key: str = None,
        max_scan: int = 100,
        max_return: int = 5,
        timeout_seconds: int = 30
    ):
        # self.api_key = api_key
        self.model = None
        self.logger = RaspaLogger()

        self.max_scan = max_scan
        self.max_return = max_return
        self.timeout_seconds = timeout_seconds

        # if api_key:
        #     import google.generativeai as genai
        #     genai.configure(api_key=api_key)
        #     self.model = genai.GenerativeModel("gemini-2.5-flash")

    # ---------------------------------
    # LLM Prompt
    # ---------------------------------
    def _get_prompt(self) -> str:
        return """
You are a materials database query generator.

Your task:
Extract MOF structural search filters from user input.

Allowed parameters ONLY:

- mofid
- mofkey
- name
- database
- vf_min, vf_max
- lcd_min, lcd_max
- pld_min, pld_max
- sa_m2g_min, sa_m2g_max
- sa_m2cm3_min, sa_m2cm3_max
- pressure_unit
- loading_unit
- limit

Rules:
1. Output ONLY valid JSON.
2. If a field is not specified, set it to null.
3. Do NOT invent unsupported properties.
4. Interpret qualitative language intelligently:
   - "high surface area" → sa_m2g_min = 1500
   - "high porosity" → vf_min = 0.6
   - "microporous" → pld_max = 2.0
   - "large pore" → lcd_min = 10.0
   and others as needed based on common MOF terminology.
6. Default limit = 20 if not specified.

Return JSON in this exact structure:

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
"""
    # ---------------------------------
    # Parse User Input → fetch kwargs
    # ---------------------------------
    # def parse_structural_filters(self, user_input: str, mock=False) -> Dict:

    #     if mock or self.model is None:
    #         print(">> [MOCK MODE] Structural filter generation")

    #         mock_data = {
    #             "vf_min": 0.6 if "porosity" in user_input.lower() else None,
    #             "sa_m2g_min": 1500 if "surface" in user_input.lower() else None,
    #             "pld_max": 2.0 if "microporous" in user_input.lower() else None,
    #             "limit": 20
    #         }

    #         query = MofdbQuery(**mock_data)
    #         return self._safe_fetch_kwargs(query)

    #     try:
    #         full_prompt = f"{self._get_prompt()}\n\nUSER INPUT: {user_input}"
    #         response = self.model.generate_content(full_prompt)

    #         clean_text = response.text.replace("```json", "").replace("```", "").strip()
    #         data = json.loads(clean_text)

    #         query = MofdbQuery(**data)
    #         return self._safe_fetch_kwargs(query)

    #     except Exception as e:
    #         print("❌ LLM parsing error:", e)
    #         return {"limit": 20}

    # ---------------------------------
    # Enforce Safe Limits
    # ---------------------------------
    def _safe_fetch_kwargs(self, query: MofdbQuery) -> Dict:
        data = {k: v for k, v in query.dict().items() if v is not None}

        if "limit" not in data:
            data["limit"] = 20
        else:
            data["limit"] = min(data["limit"], self.max_scan)

        return data

    # ---------------------------------
    # Main Retrieval Function
    # ---------------------------------
#     def retrieve(
#     self,
#     user_input: str,
#     gas: Optional[str] = None,
#     pressure: Optional[float] = None,
#     min_loading: Optional[float] = None,
#     mock=False
# ):

#         fetch_kwargs = self.parse_structural_filters(user_input, mock=mock)

#         print("🔍 Fetch kwargs:", fetch_kwargs)

#         results = []
#         scanned = 0
#         start_time = time.time()

#         try:
#             for mof in fetch(**fetch_kwargs):
#                 scanned += 1

#                 if time.time() - start_time > self.timeout_seconds:
#                     print("⏹ Timeout reached.")
#                     break

#                 if gas and pressure is not None and min_loading is not None:
#                     if not self._passes_adsorption_filter(
#                         mof, gas, pressure, min_loading
#                     ):
#                         continue

#                 results.append(mof)

#                 if len(results) >= self.max_return:
#                     break

#                 if scanned >= self.max_scan:
#                     break

#         except Exception as e:
#             print("❌ Fetch error:", e)

#         print(f"✔ Scanned {scanned} MOFs")
#         print(f"✔ Returning {len(results)} MOFs")

#         if not results:
#             print("⚠ No results found. Attempting relaxed search...")
#             results = self._relaxed_search(fetch_kwargs, gas, pressure, min_loading)

#         # 🔥 Only display table — do NOT return it
#         self._display_summary_table(results)

#         index = int(input("Enter MOF index to continue (-1 to skip):"))

#         return results[index] if 0 <= index < len(results) else None
    
    def safe_fetch(self, **kwargs):
        try:
            for mof in fetch(**kwargs, telemetry=False):
                yield mof
        except RuntimeError as e:
            if "StopIteration" in str(e):
                # harmless generator issue
                return
            else:
                raise


    
    def retrieve(
        self,
        discovery_filters: Dict,
        request_id: Optional[int] = None,
        gas: Optional[str] = None,
        pressure: Optional[float] = None,
        min_loading: Optional[float] = None,
    ):
        """
        Receives already-extracted discovery filters.
        No LLM call here anymore.
        """

        fetch_kwargs = self._safe_fetch_kwargs(
            MofdbQuery(**discovery_filters)
        )

        if request_id:
            self.logger.log_step(request_id, "discovery_started", {"filters": discovery_filters}, 0)

        # print("🔍 Fetch kwargs:", fetch_kwargs)
        print(f"\n🔎 Searching MOF database with filters:")
        for k, v in fetch_kwargs.items():
            print(f"   - {k}: {v}")


        results = []
        scanned = 0
        # start_time = time.time()

        try:
            for mof in self.safe_fetch(**fetch_kwargs):
                scanned += 1

                # Timeout protection
                # if time.time() - start_time > self.timeout_seconds:
                #     print("⏹ Timeout reached.")
                #     break

                # Optional adsorption filter
                if gas and pressure is not None and min_loading is not None:
                    if not self._passes_adsorption_filter(
                        mof, gas, pressure, min_loading
                    ):
                        continue

                results.append(mof)

                if len(results) >= self.max_return:
                    break

                if scanned >= self.max_scan:
                    break

        except Exception as e:
            print("❌ Fetch error:", e)

        print(f"✔ Scanned {scanned} MOFs")
        print(f"✔ Returning {len(results)} MOFs")

        if not results:
            print("⚠ No results found. Attempting relaxed search...")
            results = self._relaxed_search(fetch_kwargs, gas, pressure, min_loading)

        self._display_summary_table(results)

        try:
            index = int(input("Enter MOF index to continue (-1 to skip): "))
        except:
            return None

        return results[index] if 0 <= index < len(results) else None


    
    def _display_summary_table(self, mofs):

        if not mofs:
            print("No MOFs to display.")
            return

        print("\n📊 Candidate MOFs:\n")
        print(
            f"{'Index':<6} {'MOF Name':<25} "
            f"{'VF':<6} {'PLD':<6} {'LCD':<6} {'SA (m2/g)':<10}"
        )
        print("-" * 90)

        for idx, mof in enumerate(mofs):

            name = getattr(mof, "name", "N/A")
            # mofid = getattr(mof, "mofid", "N/A")
            vf = round(getattr(mof, "void_fraction", 0) or 0, 3)
            pld = round(getattr(mof, "pld", 0) or 0, 2)
            lcd = round(getattr(mof, "lcd", 0) or 0, 2)
            sa = round(getattr(mof, "sa_m2g", 0) or 0, 1)

            print(
                f"{idx:<6} {name:<25} "
                f"{vf:<6} {pld:<6} {lcd:<6} {sa:<10}"
            )

        print("\n👉 Select MOF by index.")



    # ---------------------------------
    # Adsorption Filtering
    # ---------------------------------
    def _passes_adsorption_filter(self, mof, gas, pressure, min_loading):

        if not hasattr(mof, "isotherms") or not mof.isotherms:
            return False

        for iso in mof.isotherms:
            try:
                if iso.gas.lower() == gas.lower():
                    if abs(iso.pressure - pressure) < 1e-6:
                        if iso.loading >= min_loading:
                            return True
            except:
                continue

        return False

    # ---------------------------------
    # Progressive Relaxation
    # ---------------------------------
    def _relaxed_search(self, structural_filters, gas, pressure, min_loading):

        relaxed = structural_filters.copy()

        relaxed.pop("sa_m2g_max", None)
        relaxed.pop("sa_m2g_min", None)
        relaxed.pop("pld_max", None)
        relaxed.pop("pld_min", None)

        print("🔄 Relaxed filters:", relaxed)

        results = []
        scanned = 0
        start_time = time.time()

        try:
            for mof in self.safe_fetch(**relaxed):
                scanned += 1

                if time.time() - start_time > self.timeout_seconds:
                    print("⏹ Timeout reached (relaxed).")
                    break

                if gas and pressure is not None and min_loading is not None:
                    if not self._passes_adsorption_filter(
                        mof, gas, pressure, min_loading
                    ):
                        continue

                results.append(mof)

                if len(results) >= self.max_return:
                    break

                if scanned >= self.max_scan:
                    break

        except Exception as e:
            print("❌ Relaxed fetch error:", e)

        print(f"✔ Relaxed returning {len(results)} MOFs")

        return results

