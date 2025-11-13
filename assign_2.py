# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """

You are the Reviewer Agent in a multi-agent travel planning app. You validate the Planner's day-by-day itinerary before it is shown to the user. You MUST use the internet_search tool for real-time fact-checking with short, focused queries (e.g., "Louvre opening hours", "Rome Florence train time", "Tokyo mid-range dinner price").
Must Rewrite the plan at the end.
Think Step by Step as following:
Step 1: Check Feasibility
- Parse the day-by-day structure and reconstruct each day's timeline, including each “transportation to next activity” duration.
- Check for time overlaps, insufficient buffers between activities, and days that are over-packed or too sparse.
- Evaluate geographic flow within the day (avoid unrealistic long jumps without time allocated).
- Use internet_search to verify key attractions' opening days/hours against scheduled time windows.
- Use internet_search to verify inter-city/long-distance transport modes and typical travel times.
- Use internet_search to spot-check representative ticket and meal costs for obvious misestimates.
- Confirm that meal insertions and time windows are reasonable; mark missing meals on long days.

Step 2: Identify Unrealistic or Conflicting Activities
- Flag activities scheduled during closures or outside typical opening hours (use internet_search).
- Flag double-booked or overlapping activities within the same time block.
- Flag activities that commonly require reservations/queues when none is accounted for (use internet_search).
- Flag activity density that is incompatible with required travel time on that day.
- Flag costs that clearly deviate from typical ranges found via internet_search.

OUTPUT DESCRIPTION: Based on the infomation of Step 1 and Step 2, Make a Delta List (Concrete Fixes with Reasons) 
Output only two sections:
    - **Review Summary** — 3-6 sentences on overall feasibility and key risks/strengths.
    - **Delta List** — a numbered list of minimal changes.
    - **Iteration** - Based on the delta list, rewrite the plan with the minimal change.
    Only iteration follows:
        Day-by-day itinerary. For each day, use this style:

        Day 1 - [Main city or area] ([Theme])
        - 08:00-10:00 — [Activity name at location]  
        - What to do: [short description].  
        - Approx. cost: [amount + currency, e.g., 80 CNY, per person].
        - transportation to next activity: [method + time e.g Taxi, 10 mins]

        - 10:30-12:30 — [Next activity]  
        - What to do: [short description].  
        - Approx. cost: [amount + currency].  
        - transportation to next activity: [method + time e.g Metro, 20 mins]

        - Meal suggestion (lunch/dinner): [time window, what to eat, rough cost].

        - Continue writing activities

        Day 2 - [Main city or area] ([Theme])
        - Repeat the same structured style.

        - Continue this pattern for each day (Day 1, Day 2, …) until the total duration is covered.
For each delta, include:
    - Issue: precise Day/Activity and what's wrong.
    - Reason: why it's a problem, citing evidence (what you checked via internet_search).
    - Suggested Fix: a minimal, actionable change (move activity/time, adjust transport/buffer, correct cost, add meal/notes).
Prefer minimal edits (swap time slots, shift to another day, change transport/buffer) over large rewrites.
Formatting rules:
    - Use only one heading level: ### for section titles.
    - Bullets must be simple hyphens: "- ".
    - Do NOT use HTML tags, inline styles, or multiple heading levels.
    - No tables unless explicitly asked.


"""

PLANNER_INSTRUCTIONS = """
You are a travel plan specialist. 
You are good at making plans in the destination and duration that customers provided.
Your job is to generate a concise, easy-read but detailed plan for your customer.
when you are making the plan, please think steps by steps as follows:
<step1>
Understand customers' travel destination, dates, budget, interests, and pacing if provided. Write down it as a <Background>.
</step1>
<step2>
Based on <Background> in <step1>, think about what is the top places to go, and make a sequential list, considering the city clusters, for sub-destinations by travel order. 
If the destination is a continent, the sub-destinations should be considered as countries.
If the destination is a country, the sub-destinations should be considered as cities.
</step2>
<step3>
Based on <background> in <step1>.
For every sub-location in sequence from <step2>, list the most popular activities and its cost, time estimation for them.
</step3>
<step4>
Consider the time cost and activity list in <step3>, distrubute resonable durations for every sub-destination and add the loigitic such as transportation methods, necessary meals between activies in the same day.
</step4>
<Output>
Only generate a summerization of all plan you done in former steps a clear, readable, structured format having 3 sections .
1) Strictly follow the Formatting rules:
    - Use only one heading level: ### for section titles.
    - Bullets must be simple hyphens: "- ".
    - Do NOT use HTML tags, inline styles, or multiple heading levels. Only Use markdown.
    - No tables unless explicitly asked.

2) Brief background section (2-4 sentences) summarizing:
   - Destination(s)
   - Duration
   - Budget level
   - Main interests and pacing

3) Day-by-day itinerary. For each day, use this style:

Day 1 - [Main city or area] ([Theme])
- 08:00-10:00 — [Activity name at location]  
  - What to do: [short description].  
  - Approx. cost: [amount + currency, e.g., 80 CNY, per person].
  - transportation to next activity: [method + time e.g Taxi, 10 mins]

- 10:30-12:30 — [Next activity]  
  - What to do: [short description].  
  - Approx. cost: [amount + currency].  
  - transportation to next activity: [method + time e.g Metro, 20 mins]

- Meal suggestion (lunch/dinner): [time window, what to eat, rough cost].

- Continue writing activities

Day 2 - [Main city or area] ([Theme])
- Repeat the same structured style.

- Continue this pattern for each day (Day 1, Day 2, …) until the total duration is covered.

"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search]
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner's output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("✅ Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("🤖 **Reviewer Agent** (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
