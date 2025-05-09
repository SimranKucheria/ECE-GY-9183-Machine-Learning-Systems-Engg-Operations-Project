import pytest
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
import numpy as np

captions = [
    "A black and white dog is running in a grassy garden surrounded by a white fence.",
    "A Boston Terrier is running on lush green grass in front of a white fence.",
    "A black and white dog is running through the grass.",
    "A dog runs on the green grass near a wooden fence.",
    "A Boston terrier is running in the grass."
]

def test_bleu_score(results):
    generated_texts = []
    reference_texts = []
    smoothing = SmoothingFunction().method4 
    for result in results:
        generated = result['generated_caption']
        expected = result['expected_caption']
        if isinstance(generated, np.ndarray):
            generated = generated.item()
        reference = [caption.split() for caption in expected]
        candidate = generated.split() 
        reference_texts.append(reference)
        generated_texts.append(candidate)
    bleu_score = corpus_bleu(reference_texts, generated_texts, smoothing_function=smoothing)
    assert bleu_score >= 0.5, f"BLEU Score too low: {bleu_score:.2f}"

def compute_bleu(reference_caption, generated_caption):
    reference_caption = str(reference_caption)
    generated_caption = str(generated_caption)
    reference = [reference_caption.split()]
    candidate = generated_caption.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, smoothing_function=smoothie)

def test_caption_synonym_stability(sample_image_path, get_caption):
    print("Testing synonym stability ...")
    reference_caption = get_caption(sample_image_path)
    if isinstance(reference_caption, np.ndarray):
        reference_caption = reference_caption.item()
    print(reference_caption)
    for i, test_caption in enumerate(captions, start=0):
        bleu_score = compute_bleu(reference_caption, test_caption)
        assert bleu_score > 0.15, f"BLEU score for synonym caption {i} too low: {bleu_score:.3f}"

def test_caption_meaning_change(sample_image_path, get_caption):
    print("Testing meaning change ...")
    reference_caption = get_caption(sample_image_path)
    if isinstance(reference_caption, np.ndarray):
        reference_caption = reference_caption.item()
    print(reference_caption)
    changed_caption = "A cat is sleeping on a couch."
    bleu_score = compute_bleu(reference_caption, changed_caption)
    assert bleu_score < 0.2, f"BLEU score for different meaning should be low, but got: {bleu_score:.3f}"