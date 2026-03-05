!pip install -q streamlit pyngrok langgraph langchain-groq gtts click==8.2.1
!pip install diagrams graphviz

%%writefile app.py
import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
import graphviz
import math
import tempfile

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- STATE ----------------

class GraphState(TypedDict):
    query: str
    response: str
    infra: str
    diagram: str
    question: str


# ---------------- LLM ----------------

def llm():
    return ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )


# ---------------- NODES ----------------

def architect_node(state: GraphState):

    prompt = f"""
You are a senior distributed systems architect.

Explain the system design for {state['query']}.

Include:
• core services
• request flow
• databases
• scaling strategy
• reliability

Keep it clear but realistic like a FAANG interview answer.
"""

    res = llm().invoke([("human", prompt)])

    return {"response": res.content}


def infra_node(state: GraphState):

    prompt = f"""
Explain how Redis, Kafka, and CDN would be used in a system like {state['query']}.

Explain:
• what each component does
• where it sits in architecture
• why it improves scalability
"""

    res = llm().invoke([("human", prompt)])

    return {"infra": res.content}


def diagram_node(state: GraphState):

    prompt = f"""
Create a realistic architecture flow for {state['query']}.

Return ONLY arrows like this example:

User -> CDN -> LoadBalancer -> API Gateway -> Auth Service -> Cache -> Database

If the system is complex (Uber / Spotify etc) include microservices.

Example:

User -> API Gateway -> Ride Service -> Kafka -> Matching Service -> Database
"""

    res = llm().invoke([("human", prompt)])

    return {"diagram": res.content}


def interview_node(state: GraphState):

    prompt = f"""
You are a FAANG system design interviewer.

Ask ONE challenging follow-up question about designing {state['query']}.

Example topics:
• scaling bottlenecks
• caching
• database sharding
• fault tolerance
"""

    res = llm().invoke([("human", prompt)])

    return {"question": res.content}


# ---------------- GRAPH ----------------

workflow = StateGraph(GraphState)

workflow.add_node("architect", architect_node)
workflow.add_node("infra", infra_node)
workflow.add_node("diagram", diagram_node)
workflow.add_node("interview", interview_node)

workflow.set_entry_point("architect")

workflow.add_edge("architect", "infra")
workflow.add_edge("infra", "diagram")
workflow.add_edge("diagram", "interview")
workflow.add_edge("interview", END)

app_graph = workflow.compile()


# ---------------- DIAGRAM GENERATOR ----------------

import re

def generate_dynamic_diagram(flow):

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR")

    # Split flow
    raw_nodes = [n.strip() for n in flow.split("->")]

    nodes = []
    labels = {}

    # sanitize node ids
    for i, node in enumerate(raw_nodes):

        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', node)

        if safe_id == "":
            safe_id = f"node{i}"

        nodes.append(safe_id)
        labels[safe_id] = node

    # add nodes with labels
    for n in nodes:
        dot.node(n, label=labels[n])

    # connect nodes
    for i in range(len(nodes)-1):
        dot.edge(nodes[i], nodes[i+1])

    return dot


# ---------------- SCALABILITY CALCULATOR ----------------

def scalability_calc(users):

    avg_requests_per_user = 10
    peak_multiplier = 5

    daily_requests = users * avg_requests_per_user
    peak_rps = (daily_requests / 86400) * peak_multiplier

    server_capacity = 1500

    servers_needed = math.ceil(peak_rps / server_capacity)

    return int(peak_rps), max(1, servers_needed)


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="AI System Design Mentor")

st.title("🏗️ System Design Mentor")

query = st.text_input(
    "What system are we designing?",
    placeholder="Netflix / Uber / Spotify / WhatsApp"
)

users = st.slider(
    "Expected Daily Users",
    10000,
    50000000,
    100000
)


if query:

    with st.spinner("Designing architecture..."):

        result = app_graph.invoke({
            "query": query
        })


    # -------- SYSTEM DESIGN --------

    st.subheader("System Design Explanation")
    st.write(result["response"])


    # -------- INFRA --------

    st.subheader("Infrastructure (Redis / Kafka / CDN)")
    st.write(result["infra"])


    # -------- DIAGRAM --------

    st.subheader("LLM Generated Architecture Diagram")

    diagram = generate_dynamic_diagram(result["diagram"])

    st.graphviz_chart(diagram)


    # -------- DOWNLOAD DIAGRAM --------

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

        diagram.render(tmp.name, format="png", cleanup=True)

        file_path = tmp.name + ".png"


    with open(file_path, "rb") as file:

        st.download_button(
            label="Download Architecture Diagram",
            data=file,
            file_name=f"{query}_architecture.png",
            mime="image/png"
        )


    # -------- SCALABILITY --------

    st.subheader("Scalability Calculator")

    rps, servers = scalability_calc(users)

    st.metric("Estimated Requests / sec", rps)
    st.metric("Estimated Backend Servers", servers)

    st.caption(
        """
Assumptions:
• ~10 requests per user per day  
• 5× peak traffic multiplier  
• 1500 RPS capacity per backend server
"""
    )


    # -------- INTERVIEW MODE --------

    st.subheader("System Design Interview Question")

    st.write(result["question"])


    answer = st.text_area("Your Answer")


    if st.button("Evaluate My Answer"):

        prompt = f"""
You are a FAANG system design interviewer.

Evaluate the candidate answer.

Question:
{result['question']}

Candidate Answer:
{answer}

Provide:

Score (0-10)

Strengths

Missing Points

Improvement Suggestions
"""

        feedback = llm().invoke([("human", prompt)])

        st.subheader("Interview Feedback")
        st.write(feedback.content)

!GROQ_API_KEY="GROQ_API_KEY" streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > log.txt 2>&1 &

import time
time.sleep(10)

from pyngrok import ngrok

ngrok.kill()

ngrok.set_auth_token("NGROK_API_KEY")

tunnel = ngrok.connect(8501, "http")

print("🚀 OPEN THIS LINK:")
print(tunnel.public_url)
