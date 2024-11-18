from flask import Flask, render_template
from core.loaders import DocumentLoader, DialogueLoader, DPODialogueLoader
from core.components import PedagogicalRules

app = Flask(__name__)
base_path = "/home/gp1108/Code/Thesis/dataset_generation/data"
document_loader = DocumentLoader(f"{base_path}/extracted_texts.json")
dialogue_loader = DialogueLoader(f"{base_path}/dialogues.json")
dpo_dialogue_loader = DPODialogueLoader(f"{base_path}/dpo_dialogues.json")
rules = PedagogicalRules(f"/home/gp1108/Code/Thesis/dataset_generation/prompts/rules.txt")

@app.route("/")
def home():
    documents = [doc.id for doc in document_loader]
    dialogues = [dlg.id for dlg in dialogue_loader]
    dpo_dialogues = dpo_dialogue_loader.get_unique_dpo_ids()

    return render_template("home.html",
                           documents=documents,
                           dialogues=dialogues,
                           dpo_dialogues=dpo_dialogues)

@app.route('/document/<doc_id>')
def document_page(doc_id):
    document = document_loader.get_document_by_id(doc_id)
    dialogues = dialogue_loader.get_dialogues_by_document_id(doc_id)
    return render_template('document.html',
                           document=document,
                           dialogues=dialogues)

@app.route('/chunk/<chunk_id>')
def chunk_page(chunk_id):
    # Getting the document id from the chunk id
    document_id = chunk_id.split("_")[0]
    document = document_loader.get_document_by_id(document_id)
    chunk = document.get_chunk_by_id(chunk_id)
    if not chunk:
        return "Chunk not found", 404

    return render_template('chunk.html', chunk=chunk, document=document)

@app.route('/dialogue/<dialogue_id>')
def dialogue_page(dialogue_id):
    dialogue = dialogue_loader.get_dialogue_by_id(dialogue_id)

    if not dialogue:
        return "Dialogue not found", 404

    # Extract document ID and chunk IDs
    doc_id = dialogue.id.split("_ch")[0]
    chunk_ids = [f"{doc_id}_ch{chunk}" for chunk in dialogue.id.split("[")[1].strip("]").split("_")]

    dpo_dialogues = dpo_dialogue_loader.get_dpo_dialogues_by_dialogue_id(dialogue_id)

    return render_template('dialogue.html',
                           dialogue=dialogue,
                           doc_id=doc_id,
                           chunk_ids=chunk_ids,
                           dpo_dialogues=dpo_dialogues)

@app.route('/dpo_dialogue/<dpo_id>')
def dpo_dialogue_page(dpo_id):
    dialogue = dpo_dialogue_loader.get_dpo_dialogue_by_id(dpo_id)
    
    if not dialogue:
        return "DPO Dialogue not found", 404

    # Get all turns for this DPO dialogue
    turns = dpo_dialogue_loader.get_dpo_turns_by_dialogue_id(dpo_id)
    
    # Get involved chunks and document
    chunk_ids = dialogue.get_chunks_ids()

    # Get original dialogue
    original_diag_id = dialogue.id.split("_dpo")[0]

    doc_id = dialogue.get_doc_id()

    return render_template('dpo_dialogue.html',
                           dialogue=dialogue,
                           doc_id=doc_id,
                           chunk_ids=chunk_ids,
                           turns=turns,
                           original_diag_id=original_diag_id,
                           rules=rules)

if __name__ == "__main__":
    app.run(debug=True)
