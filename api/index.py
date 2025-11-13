from fastapi import FastAPI
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import uuid
import os
import json # New import
from google.oauth2 import service_account # New import
from datetime import datetime, timedelta

# This creates your web application
app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "API is running", "message": "Use POST /generate"}

@app.get("/routes")
def list_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "methods": route.methods if hasattr(route, 'methods') else []
        })
    return {"routes": routes}



# --- 1. CONFIGURATION ---
# These are now read from Vercel's Environment Variables
PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
LOCATION = "us-central1"

# --- LOAD SERVICE ACCOUNT CREDENTIALS ---
# This reads the secret JSON key from Vercel's environment
GCP_SA_KEY_STRING = os.environ.get("GCP_SA_KEY")
if not GCP_SA_KEY_STRING:
    raise ValueError("GCP_SA_KEY environment variable is not set.")

# Convert the string key into credentials
try:
    GCP_CREDENTIALS = service_account.Credentials.from_service_account_info(
        json.loads(GCP_SA_KEY_STRING)
    )
except json.JSONDecodeError:
    raise ValueError("GCP_SA_KEY is not a valid JSON string.")

# --- 2. INITIALIZE SERVICES ---
# We now initialize everything with our new credentials
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=GCP_CREDENTIALS)
storage_client = storage.Client(project=PROJECT_ID, credentials=GCP_CREDENTIALS)
output_bucket = storage_client.bucket(BUCKET_NAME)

# Load the models
image_model = GenerativeModel(model_name="gemini-2.5-flash-image")
text_model = GenerativeModel(model_name="gemini-2.5-pro") 

# --- 3. DEFINE YOUR FOUR "TOOLS" ---

def generate_chibi_image(prompt: str) -> dict:
    """Generates an image in the 'elsa_chibi_face' style."""
    print(f"Tool Call: generate_chibi_image('{prompt}')...")
    CHIBI_STYLE_GUIDE_URIS = [ f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_01.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_02.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_03.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_04.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_10.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_15.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_09_dollar.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_09_heart.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_09_sad.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_13_wink.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_13_star.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_13_sleep.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_13_angry.png", f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_20_cropped.png" ]
    prompt_parts = [f"Using the {len(CHIBI_STYLE_GUIDE_URIS)} uploaded images as a visual style reference for the 'chibi mascot' character, fulfill this request: '{prompt}'"]
    prompt_parts.extend([Part.from_uri(uri, mime_type="image/png") for uri in CHIBI_STYLE_GUIDE_URIS])
    try:
        response = image_model.generate_content(prompt_parts)
        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                break
        if image_bytes is None: raise Exception("Model did not return an image.")
        file_name = f"outputs/chibi_{uuid.uuid4()}.png"
        blob = output_bucket.blob(file_name)
        blob.upload_from_string(image_bytes, content_type="image/png")
        blob.make_public() # This works because your bucket is "Fine-grained"
        return {"type": "image", "content": blob.public_url}
    except Exception as e:
        return {"type": "text", "content": f"Error generating Chibi image: {e}"}

def generate_robot_image(prompt: str) -> dict:
    """Generates an image in the 'elsa_robot' (red robot) style."""
    print(f"Tool Call: generate_robot_image('{prompt}')...")
    ROBOT_STYLE_GUIDE_URIS = [ f"gs://{BUCKET_NAME}/robot_images/elsa_05.png", f"gs://{BUCKET_NAME}/robot_images/elsa_06.png", f"gs://{BUCKET_NAME}/robot_images/elsa_07.png", f"gs://{BUCKET_NAME}/robot_images/elsa_08.png", f"gs://{BUCKET_NAME}/robot_images/elsa_11.png", f"gs://{BUCKET_NAME}/robot_images/elsa_12.png", f"gs://{BUCKET_NAME}/robot_images/elsa_17.png", f"gs://{BUCKET_NAME}/robot_images/elsa_18.png", f"gs://{BUCKET_NAME}/robot_images/elsa_19.png" ]
    prompt_parts = [f"Using the {len(ROBOT_STYLE_GUIDE_URIS)} uploaded images as a visual style reference for the 'red robot' character, fulfill this request: '{prompt}'"]
    prompt_parts.extend([Part.from_uri(uri, mime_type="image/png") for uri in ROBOT_STYLE_GUIDE_URIS])
    try:
        response = image_model.generate_content(prompt_parts)
        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                break
        if image_bytes is None: raise Exception("Model did not return an image.")
        file_name = f"outputs/robot_{uuid.uuid4()}.png"
        blob = output_bucket.blob(file_name)
        blob.upload_from_string(image_bytes, content_type="image/png")
        blob.make_public()
        return {"type": "image", "content": blob.public_url}
    except Exception as e:
        return {"type": "text", "content": f"Error generating Robot image: {e}"}

def generate_general_image(prompt: str) -> dict:
    """Generates a general image with NO style guide."""
    print(f"Tool Call: generate_general_image('{prompt}')...")
    try:
        response = image_model.generate_content([prompt]) 
        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                break
        if image_bytes is None: raise Exception("Model did not return an image.")
        file_name = f"outputs/general_{uuid.uuid4()}.png"
        blob = output_bucket.blob(file_name)
        blob.upload_from_string(image_bytes, content_type="image/png")
        blob.make_public()
        return {"type": "image", "content": blob.public_url}
    except Exception as e:
        return {"type": "text", "content": f"Error generating general image: {e}"}

def generate_diagram_code(topic: str) -> dict:
    """Generates Mermaid.js flowchart code for a specific topic."""
    print(f"Tool Call: generate_diagram_code('{topic}')...")
    system_instruction = "You are a diagramming expert. Your only job is to generate valid Mermaid.js code for the user's request. Respond ONLY with the Mermaid.js code block, starting with ```mermaid and ending with ```. Do not add any other explanatory text."
    diagram_model = GenerativeModel(model_name="gemini-2.5-pro", system_instruction=system_instruction)
    try:
        response = diagram_model.generate_content(topic)
        return {"type": "diagram", "content": response.text}
    except Exception as e:
        return {"type": "text", "content": f"Error generating diagram: {e}"}

# --- 4. THE "AGENT BRAIN" ---

def get_agent_decision(user_prompt: str) -> (str, str):
    """
    This function acts as the "agent brain."
    It decides which tool to use.
    """
    print(f"\nAgent is thinking about: '{user_prompt}'...")
    system_instruction = (
        "You are a routing agent. Your job is to analyze the user's prompt and "
        "decide which tool is the correct one to use. "
        "You must respond with a JSON object like: {\"tool\": \"...\"}"
        "\n"
        "Here are the four tools:"
        "1. 'chibi': Use this if the prompt mentions 'chibi', '2D face', or the 'chibi mascot'."
        "2. 'robot': Use this if the prompt mentions 'red robot', '3D robot', or the 'robot mascot'."
        "3. 'diagram': Use this for any 'flowchart', 'diagram', 'schematic', or general text question."
        "4. 'general': Use this ONLY if the prompt asks for an image but does NOT mention 'chibi', 'robot', or 'diagram'."
        "\n"
        "Prioritize the mascots. If a complex prompt (like a 'banner' or 'scene') mentions 'chibi', "
        "the tool MUST be 'chibi'. 'general' is only a fallback."
    )
    routing_model = GenerativeModel(model_name="gemini-2.5-pro", system_instruction=system_instruction)
    try:
        response = routing_model.generate_content(user_prompt)
        import json
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        decision = json.loads(json_text)
        tool = decision.get("tool", "general").lower()
        if tool not in ['chibi', 'robot', 'diagram', 'general']:
            tool = 'general'
        print(f"Agent decided to use tool: '{tool}'")
        return tool, user_prompt
    except Exception as e:
        print(f"Error in agent brain: {e}. Defaulting to 'general'.")
        return 'general', user_prompt

# --- 5. THE API ENDPOINT ---
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def handle_generation(request: PromptRequest):
    """
    This is the main API endpoint.
    It receives a prompt, asks the "brain" what to do,
    and then calls the correct tool.
    """
    tool_to_use, task_prompt = get_agent_decision(request.prompt)
    
    result = {}
    if tool_to_use == 'chibi':
        result = generate_chibi_image(task_prompt)
    elif tool_to_use == 'robot':
        result = generate_robot_image(task_prompt)
    elif tool_to_use == 'diagram':
        result = generate_diagram_code(task_prompt)
    elif tool_to_use == 'general':
        result = generate_general_image(task_prompt)
    
    return result
