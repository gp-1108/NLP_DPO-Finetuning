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

if __name__ == "__main__":
    app.run(debug=True)