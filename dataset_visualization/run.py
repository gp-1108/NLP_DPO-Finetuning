from flask import Flask, render_template
from core.loaders import DocumentLoader, DialogueLoader, DPODialogueLoader

app = Flask(__name__)
base_path = "/home/gp1108/Code/Thesis/dataset_generation/data"
document_loader = DocumentLoader(f"{base_path}/extracted_texts.json")
dialogue_loader = DialogueLoader(f"{base_path}/dialogues.json")
dpo_dialogue_loader = DPODialogueLoader(f"{base_path}/dpo_dialogues.json")

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

@app.route('/chunks/<chunk_id>')
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

if __name__ == "__main__":
    app.run(debug=True)
