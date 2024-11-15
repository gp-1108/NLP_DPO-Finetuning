from flask import Flask, render_template
from core.loaders import DocumentLoader, DialogueLoader, DPODialogueLoader

app = Flask(__name__)

@app.route("/")
def home():
    base_path = "/home/gp1108/Code/Thesis/dataset_generation/data"
    document_loader = DocumentLoader(f"{base_path}/extracted_texts.json")
    dialogue_loader = DialogueLoader(f"{base_path}/dialogues.json")
    dpo_dialogue_loader = DPODialogueLoader(f"{base_path}/dpo_dialogues.json")

    documents = [doc.id for doc in document_loader]
    dialogues = [dlg.id for dlg in dialogue_loader]
    dpo_dialogues = dpo_dialogue_loader.get_unique_dpo_ids()

    return render_template("home.html",
                           documents=documents,
                           dialogues=dialogues,
                           dpo_dialogues=dpo_dialogues)

if __name__ == "__main__":
    app.run(debug=True)
