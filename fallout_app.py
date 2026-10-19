import streamlit as st
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Load .env (OpenAI + LangSmith keys)
load_dotenv()

st.set_page_config(
    page_title="Order Fallout Triage (Human-in-the-Loop)",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Define Graph State & Structured Output Schema
# ---------------------------------------------------------
class FalloutState(TypedDict):
    order_id: str
    error_code: str
    order_details: str
    root_cause: str
    risk_level: str          # "LOW" or "HIGH"
    proposed_action: str
    resolution_status: str

class TriageDecision(BaseModel):
    """Structured diagnosis of an order fallout ticket."""
    root_cause: str = Field(description="Clear 1-sentence technical explanation of why the order failed.")
    risk_level: Literal["LOW", "HIGH"] = Field(
        description="LOW if it is a formatting/address validation sync issue. HIGH if it requires physical port/inventory database reassignment."
    )
    proposed_action: str = Field(description="Exact API or database action to resolve the fallout.")

# ---------------------------------------------------------
# 3. Build & Cache the Stateful Graph with MemorySaver
# ---------------------------------------------------------
@st.cache_resource
def build_triage_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_triage = llm.with_structured_output(TriageDecision)

    triage_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Senior Telecom Order-to-Activate Fallout Triage Agent. "
         "Analyze the failed order payload.\n"
         "- Mark risk_level as 'LOW' for minor address formatting/GIS postal sync mismatches that can be auto-corrected.\n"
         "- Mark risk_level as 'HIGH' for physical fiber port exhaustion or splitter inventory mismatches that require a live database port override."),
        ("human", "Order ID: {order_id}\nError Code: {error_code}\nDetails: {order_details}")
    ])
    triage_chain = triage_prompt | structured_triage

    # --- Node 1: AI Diagnosis & Risk Classification ---
    def diagnose_fallout_node(state: FalloutState):
        decision = triage_chain.invoke({
            "order_id": state["order_id"],
            "error_code": state["error_code"],
            "order_details": state["order_details"]
        })
        return {
            "root_cause": decision.root_cause,
            "risk_level": decision.risk_level,
            "proposed_action": decision.proposed_action,
            "resolution_status": "DIAGNOSED"
        }

    # --- Node 2a: Low-Risk Auto Resolution ---
    def auto_resolve_node(state: FalloutState):
        return {
            "resolution_status": (
                f"✅ AUTO-RESOLVED (Zero-Touch): Executed '{state['proposed_action']}' "
                f"for Order {state['order_id']}. Order pushed to Activation Queue."
            )
        }

    # --- Node 2b: High-Risk Database Override (Protected by HITL Interrupt!) ---
    def execute_override_node(state: FalloutState):
        return {
            "resolution_status": (
                f"🔐 HUMAN-APPROVED OVERRIDE EXECUTED: Performed '{state['proposed_action']}' "
                f"for Order {state['order_id']}. Secondary fiber port assigned & activated!"
            )
        }

    # --- Conditional Edge Router ---
    def route_by_risk(state: FalloutState):
        if state["risk_level"] == "HIGH":
            return "execute_override"
        return "auto_resolve"

    # --- Assemble Graph ---
    workflow = StateGraph(FalloutState)
    workflow.add_node("diagnose_fallout", diagnose_fallout_node)
    workflow.add_node("auto_resolve", auto_resolve_node)
    workflow.add_node("execute_override", execute_override_node)

    workflow.add_edge(START, "diagnose_fallout")
    workflow.add_conditional_edges(
        "diagnose_fallout",
        route_by_risk,
        {
            "auto_resolve": "auto_resolve",
            "execute_override": "execute_override"
        }
    )
    workflow.add_edge("auto_resolve", END)
    workflow.add_edge("execute_override", END)

    # CRITICAL: MemorySaver + interrupt_before pauses execution before 'execute_override' runs!
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["execute_override"]
    )
    return app

app = build_triage_graph()

# ---------------------------------------------------------
# 4. Sidebar: Visual Architecture Diagram
# ---------------------------------------------------------
with st.sidebar:
    st.header("🗺️ HITL Graph Topology")
    try:
        st.image(app.get_graph().draw_mermaid_png(), caption="Notice the interrupt before execute_override")
    except Exception:
        st.info("Mermaid diagram preview unavailable.")
    st.markdown("""
    **Core Primitives Used:**
    - `MemorySaver`: Persists thread state across turns.
    - `interrupt_before`: Freezes graph before high-risk actions.
    - `with_structured_output`: Forces deterministic `LOW` vs `HIGH` risk routing.
    """)

# ---------------------------------------------------------
# 5. Main UI: Sample Fallout Tickets & Execution
# ---------------------------------------------------------
st.title("🛡️ Order Fallout Triage Agent (Human-in-the-Loop)")
st.markdown(
    "This agent triages failed provisioning orders. **Low-risk** address issues resolve automatically, "
    "while **High-risk** inventory overrides pause at a **LangGraph `interrupt_before` gate** for human approval."
)

# Preset Scenarios
col1, col2 = st.columns(2)
with col1:
    if st.button("🚨 Scenario 1: High-Risk Inventory Fallout (Triggers Pause Gate)", use_container_width=True):
        st.session_state["order_id"] = "ORD-99412-FIBER"
        st.session_state["error_code"] = "ERR_INV_PORT_EXHAUSTED_409"
        st.session_state["order_details"] = (
            "Customer activation failed at Cabinet OLT-04. Primary Splitter Port #7 shows "
            "status 'RESERVED_FAULTY' in inventory DB. Secondary Port #8 is available."
        )
        st.session_state["thread_id"] = "thread_high_risk_01"

with col2:
    if st.button("⚡ Scenario 2: Low-Risk Address Fallout (Zero-Touch Auto-Fix)", use_container_width=True):
        st.session_state["order_id"] = "ORD-88201-FIBER"
        st.session_state["error_code"] = "ERR_GIS_ADDR_FORMAT_102"
        st.session_state["order_details"] = (
            "Order entry has street suffix 'Ave.' while GIS Master Database expects 'Avenue'. "
            "Serviceable location ID matches 100%."
        )
        st.session_state["thread_id"] = "thread_low_risk_02"

# Input Fields
order_id = st.text_input("Order ID:", value=st.session_state.get("order_id", "ORD-99412-FIBER"))
error_code = st.text_input("Fallout Error Code:", value=st.session_state.get("error_code", "ERR_INV_PORT_EXHAUSTED_409"))
order_details = st.text_area(
    "Fallout Payload Details:",
    value=st.session_state.get(
        "order_details",
        "Customer activation failed at Cabinet OLT-04. Primary Splitter Port #7 shows status 'RESERVED_FAULTY' in inventory DB. Secondary Port #8 is available."
    )
)
thread_id = st.session_state.get("thread_id", "thread_high_risk_01")
config = {"configurable": {"thread_id": thread_id}}

if st.button("🚀 Run Triage Agent", type="primary"):
    initial_input = {
        "order_id": order_id,
        "error_code": error_code,
        "order_details": order_details
    }
    with st.spinner("Diagnosing order fallout..."):
        for _ in app.stream(initial_input, config=config):
            pass

# ---------------------------------------------------------
# 6. Inspect Graph Checkpoint State (Detect If Paused!)
# ---------------------------------------------------------
current_snapshot = app.get_state(config)

if current_snapshot.values:
    state_vals = current_snapshot.values
    st.divider()
    st.subheader(f"📋 Triage State Inspector (Thread: `{thread_id}`)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Order ID", state_vals.get("order_id"))
    m2.metric("Error Code", state_vals.get("error_code"))
    m3.metric("AI Risk Classification", state_vals.get("risk_level"))

    st.info(f"**🔍 Root Cause Diagnosis:** {state_vals.get('root_cause')}")
    st.warning(f"**🛠️ Proposed Action:** `{state_vals.get('proposed_action')}`")

    # Check if LangGraph is paused at 'execute_override'
    if "execute_override" in current_snapshot.next:
        st.error(
            "⏸️ **LANGGRAPH INTERRUPT GATE ACTIVE:** Execution is paused BEFORE node `execute_override`. "
            "Waiting for Supervisor approval to modify live inventory."
        )

        # Allow the human supervisor to edit the proposed action before resuming!
        edited_action = st.text_input(
            "✏️ Optional Supervisor Edit to Proposed Action before execution:",
            value=state_vals.get("proposed_action")
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("✅ Approve & Resume Graph Execution", type="primary", use_container_width=True):
                # Update state with human edits (if any) and pass None to resume from checkpoint!
                app.update_state(config, {"proposed_action": edited_action})
                for _ in app.stream(None, config=config):
                    pass
                st.rerun()

        with b2:
            if st.button("❌ Reject & Escalate to Field Ops", use_container_width=True):
                app.update_state(
                    config,
                    {"resolution_status": "🚫 REJECTED BY SUPERVISOR: Ticket routed to Field Engineering queue."},
                    as_node="execute_override"  # Pretend node finished so graph exits cleanly to END
                )
                st.rerun()

    else:
        # Graph reached END
        status_text = state_vals.get("resolution_status", "")
        if "REJECTED" in status_text:
            st.error(status_text)
        else:
            st.success(status_text)