# from intent_extractor import IntentExtractor
# from mof_query_agent import MofDiscoveryAgent
# from raspa_agent import RaspaAgent

# class WorkflowEngine:

#     def __init__(self, api_key=None):
#         self.extractor = IntentExtractor(api_key)
#         self.discovery = MofDiscoveryAgent(api_key)
#         self.raspa = RaspaAgent(api_key)

#     def run(self, user_input):

#         # 1️⃣ Extract intent
#         intent = self.extractor.extract(user_input)

#         discovery_part = intent["discovery"]
#         simulation_part = intent["simulation"]

#         # 2️⃣ Retrieve MOFs
#         selected_mof = self.discovery.retrieve(
#             user_input,
#             mock=False
#         )

#         if selected_mof is None:
#             print("No MOF selected.")
#             return

#         # 3️⃣ Inject framework into simulation
#         simulation_part["system"]["framework_name"] = selected_mof.name

#         # 4️⃣ Convert to RaspaRequest
#         raspa_request = self.raspa.parse_with_llm(
#             user_input,
#             mock=False
#         )

#         raspa_request.system.framework_name = selected_mof.name

#         # 5️⃣ Generate config
#         config = self.raspa.merge_defaults(raspa_request)

#         # 6️⃣ Generate file
#         content = self.raspa.generate_raspa_file_content(config)

#         with open("simulation.input", "w") as f:
#             f.write(content)

#         print("Simulation file generated.")

from intent_extractor import IntentExtractor
from mof_query_agent import MofDiscoveryAgent
from raspa_agent import RaspaAgent
from raspa_logger import RaspaLogger
import time


class WorkflowEngine:

    def __init__(self, api_key=None):
        self.extractor = IntentExtractor(api_key)
        self.discovery = MofDiscoveryAgent()   # no LLM here anymore
        self.raspa = RaspaAgent()              # no LLM here anymore
        self.logger = RaspaLogger()

    def run(self, user_input: str, mock: bool = False):
        start_time = time.time()
        request_id = self.logger.log_request({"query": user_input})

        # ----------------------------------
        # 1️⃣ Extract Unified Intent
        # ----------------------------------
        print("🧠 Extracting intent...")
        step_start = time.time()
        intent = self.extractor.extract(user_input, request_id=request_id, mock=mock)
        self.logger.log_step(request_id, "intent_extracted", intent, time.time() - step_start)

        discovery_block = intent["discovery"]
        simulation_block = intent["simulation"]

        # ----------------------------------
        # 2️⃣ MOF Discovery
        # ----------------------------------
        print("\n🔍 Retrieving candidate MOFs...")
        step_start = time.time()
        selected_mof = self.discovery.retrieve(discovery_block, request_id=request_id)
        
        if selected_mof is None:
            self.logger.log_step(request_id, "mof_retrieved", {"status": "none found"}, time.time() - step_start)
            self.logger.log_final_result(request_id, "Workflow terminated: No MOF selected")
            print("❌ No MOF selected. Workflow terminated.")
            return

        self.logger.log_step(request_id, "mof_retrieved", {"mof_name": selected_mof.name}, time.time() - step_start)
        print(f"\n✔ Selected MOF: {selected_mof.name}")

        # ----------------------------------
        # 3️⃣ Run RASPA Simulation
        # ----------------------------------
        print("\n⚙ Preparing RASPA simulation...")
        step_start = time.time()
        self.raspa.run(
            simulation_block=simulation_block,
            selected_framework=selected_mof.name,
            request_id=request_id
        )
        self.logger.log_step(request_id, "raspa_completed", {"status": "success"}, time.time() - step_start)

        self.logger.log_final_result(request_id, "Workflow completed successfully")

