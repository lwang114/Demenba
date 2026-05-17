from nemo.collections.asr.models import SortformerEncLabelModel

# load model from Hugging Face model card directly (You need a Hugging Face token)
diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")

# switch to inference mode
diar_model.eval()

audio_input='./data/test.jsonl'
out_path = './data/test_diarized.tsv'
predicted_segments = diar_model.diarize(audio=audio_input, batch_size=1)
print('predicted_segments:', predicted_segments)  # XXX
#for start, end, spk in predicted_segments:
#    print()

