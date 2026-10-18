"""
Fixed & runnable version of multi_agent_ecosistem.ipynb
- Works outside Google Colab (uses .env / environment variable)
- Correct relative paths
- Minor robustness improvements in speaker selection and tool registration
- Requires: GROQ_API_KEY in environment or .env file
"""

import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

import autogen
from autogen import UserProxyAgent, register_function, AssistantAgent

# ---------------------------------------------------------------------------
# 1. Model configuration
# ---------------------------------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a free key at https://console.groq.com "
        "and put it in a .env file or export it."
    )

llama31_config = {
    "config_list": [
        {
            "model": "llama-3.1-8b-instant",
            "api_key": api_key,
            "api_type": "groq",
        }
    ]
}

# Uses the currently available Groq model (was labelled gemma2 in the original markdown)
gemma2_config = {
    "config_list": [
        {
            "model": "openai/gpt-oss-120b",
            "api_key": api_key,
            "api_type": "groq",
        }
    ]
}

# ---------------------------------------------------------------------------
# 2. Termination policy
# ---------------------------------------------------------------------------
def termination_message(msg) -> bool:
    content = (msg.get("content") or "").lower()
    return "terminate" in content

# ---------------------------------------------------------------------------
# 3. Agents
# ---------------------------------------------------------------------------
user_proxy = UserProxyAgent(
    name="User",
    llm_config=False,
    is_termination_msg=termination_message,
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

cv_analyser = AssistantAgent(
    name="CV Analyser",
    description="Analyses the input CV and returns the most relevant information.",
    system_message=(
        "You must return an analysis of the input file.\n"
        "Other details for your answer:\n"
        "- Do not respond any piece of code.\n"
        "- Do not recommend courses; just analyse the input file."
    ),
    llm_config=llama31_config,
)

skills_extractor = AssistantAgent(
    name="Skills Extractor",
    description="Returns a list of the candidate's skills",
    system_message=(
        "Your task is to extract the main skills from the input received.\n"
        "You must return the skills as a comma-separated list. "
        "For example: 'Python, Java, SQL'. Do not add more information, "
        "just the list of skills in the specified format.\n"
        "Other details for your answer:\n"
        "- Do not respond any piece of code.\n"
        "- Do not recommend courses; just return the skills as a comma-separated list."
    ),
    llm_config=llama31_config,
)

courses_recommender = AssistantAgent(
    name="Courses recommender",
    description=(
        "Recommend courses. Read the courses from data/courses.txt "
        "and return the courses that match the skills of the input file."
    ),
    system_message=(
        "The goal is to recommend courses. Follow these steps:\n"
        "1) Read the list of courses from data/courses.txt (use the read_txt_courses tool).\n"
        "2) Combine the list of courses with the profile of the candidate.\n"
        "3) Return the courses that match the skills of the candidate.\n"
        "Do not respond any piece of code."
    ),
    llm_config=llama31_config,
)

resolution_checker = AssistantAgent(
    name="Resolution Checker Agent",
    description=(
        "Checks if the user's query has been resolved. "
        "If resolved, respond with 'TERMINATE' + the final answer; otherwise respond with 'CONTINUE'."
    ),
    system_message=(
        "You should respond with 'TERMINATE' if the query has been resolved. "
        "Otherwise respond with 'CONTINUE'.\n"
        "Try to be resolute: do not reply 'TERMINATE' if the initial query has not been resolved.\n"
        "**IMPORTANT**: Only respond with 'TERMINATE' when the query is fully resolved. "
        "When you answer 'TERMINATE' you MUST also generate a complete final answer to the "
        "user's query that synthesises all information given by the other agents."
    ),
    llm_config=gemma2_config,
    is_termination_msg=termination_message,
)

# ---------------------------------------------------------------------------
# 4. Tools
# ---------------------------------------------------------------------------
def read_txt(path: str) -> str:
    """Reads a text file and returns its content as a string (UTF-8)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {e}"

register_function(
    read_txt,
    caller=cv_analyser,
    executor=user_proxy,
    name="read_txt_cv_analyse",
    description="Read the profile / CV of the user",
)

register_function(
    read_txt,
    caller=courses_recommender,
    executor=user_proxy,
    name="read_txt_courses",
    description="Read available courses and/or the profile of the user",
)

register_function(
    read_txt,
    caller=skills_extractor,
    executor=user_proxy,
    name="read_txt_cv_skills",
    description="Read the profile / CV of the user",
)

# ---------------------------------------------------------------------------
# 5. Custom hybrid speaker selection
# ---------------------------------------------------------------------------
def custom_speaker_selection(last_speaker, groupchat):
    agents_to_check = ["cv analyser", "skills extractor", "courses recommender"]

    last_message = groupchat.messages[-1]
    penultimate_message = groupchat.messages[-2] if len(groupchat.messages) > 1 else None

    last_speaker_name = last_speaker.name.lower()
    penultimate_speaker_name = (
        penultimate_message["name"].lower() if penultimate_message else None
    )

    # Tool call → User (executor)
    if last_message.get("tool_calls"):
        return next(a for a in groupchat.agents if a.name == "User")

    # After User executed a tool, give control back to the agent that requested it
    if (
        last_speaker_name == "user"
        and penultimate_speaker_name
        and penultimate_speaker_name not in ["user", "resolution checker agent"]
    ):
        return next(
            a for a in groupchat.agents if a.name.lower() == penultimate_speaker_name
        )

    # After a specialist agent finishes → Resolution Checker
    if last_speaker_name in agents_to_check:
        return next(
            a for a in groupchat.agents if a.name == "Resolution Checker Agent"
        )

    # Fall back to LLM-based selection
    return "auto"

# ---------------------------------------------------------------------------
# 6. Group chat + manager
# ---------------------------------------------------------------------------
groupchat = autogen.GroupChat(
    agents=[
        user_proxy,
        cv_analyser,
        skills_extractor,
        courses_recommender,
        resolution_checker,
    ],
    messages=[],
    allow_repeat_speaker=True,
    speaker_selection_method=custom_speaker_selection,
    role_for_select_speaker_messages="system",
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    code_execution_config={"use_docker": False},
    llm_config=gemma2_config,
    is_termination_msg=termination_message,
)

# ---------------------------------------------------------------------------
# 7. Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cv_file_path = "data/cv.txt"
    message = f"Recommend courses for the profile {cv_file_path}"

    chat_result = user_proxy.initiate_chat(
        recipient=manager,
        message=message,
        max_turns=30,
    )

    print("\n===== AGENT ORDER =====")
    print([x["name"] for x in chat_result.chat_history])

    print("\n===== FINAL RESULT =====")
    print(chat_result.chat_history[-1]["content"])
