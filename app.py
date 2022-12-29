from flask import Flask, render_template, request
from transformers import AutoTokenizer
import pickle

app = Flask(__name__)

models = []
name = []
tokrnizers = []

model1 = pickle.load(open("t5-small-Text2HL(xlsum(eng))-adapter.pkl", "rb"))
model2 = pickle.load(open("t5-small-Summary2HL(xlsum(eng))-adapter.pkl", "rb"))

tokenizer1 = AutoTokenizer.from_pretrained('t5-small')

name.append('T5-small-HL-Adapter(from Text)')
models.append(model1)
tokrnizers.append(tokenizer1)

name.append('T5-small-HL-Adapter(from summary)')
models.append(model1)
tokrnizers.append(tokenizer1)


# ===========================================================================================================================
# ===========================================================================================================================


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inputlist = []
        txt = request.form['txt']
        txt = 'summarize: ' + txt
        inputlist.append(txt)
        summary = request.form['sum']
        summary = 'summarize: ' + summary
        inputlist.append(summary)
        results = []

        for inputtxt, model, tokenizer in zip(inputlist, models, tokrnizers):
            inputs = tokenizer(inputtxt, return_tensors="pt",
                               truncation=True, padding=True)
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=True,  # disable sampling to test if batching affects output,
                min_length=10,
                max_length=30,
                early_stopping=True
            )

            final_output = tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True)
            results.append(final_output)

        info = zip(name, results)
        return render_template('output.html', info=info, text=txt, summary=summary)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
