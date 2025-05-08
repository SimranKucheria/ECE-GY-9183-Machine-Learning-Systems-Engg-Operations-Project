import pytest
import requests
from nltk.translate.bleu_score import sentence_bleu


def test_bleu_score_above_threshold(test_data):
    threshold = 0.3  # Adjust based on your domain needs
    for sample in test_data:
        generated = generate_caption_triton(sample['image_path'])
        reference = [sample['expected_caption'].split()]
        candidate = generated.split()
        score = sentence_bleu(reference, candidate)
        print(f"BLEU score for {sample['image_path']}: {score}")
        assert score >= threshold


# def test_caption_template(test_data):
#     for sample in test_data:
#         generated = generate_caption_triton(sample['image_path'])
#         assert generated.startswith(('A ', 'An ')), f"Caption does not follow template: {generated}"


# def test_known_failures():
#     known_failure_image = "path/to/complex_scene.jpg"
#     expected_keywords = ['crowded', 'market']
#     generated = generate_caption_triton(known_failure_image)
#     contains_keyword = any(keyword in generated for keyword in expected_keywords)
#     pytest.xfail("Known issue: Model struggles with crowded scenes.")
#     assert contains_keyword


# def test_caption_is_not_empty(test_data):
#     for sample in test_data:
#         generated = generate_caption_triton(sample['image_path'])
#         assert isinstance(generated, str) and generated.strip() != ""


# def test_caption_length_reasonable(test_data):
#     max_words = 30
#     for sample in test_data:
#         generated = generate_caption_triton(sample['image_path'])
#         num_words = len(generated.split())
#         assert num_words <= max_words, f"Caption too long: {num_words} words"