from clever.phase2_code_retriever.pipeline import KCDEvaluator


def test_original_phase2_metric_exact_and_main_code_match():
    evaluator = KCDEvaluator.__new__(KCDEvaluator)

    predicted = {"record_1": ["C34.19", "R50.99"]}
    gold = {"record_1": ["C34.11", "R50.99"]}

    results = evaluator.evaluate_predictions(predicted, gold)

    assert results["overall"]["Exact_Match"]["tp"] == 1
    assert results["overall"]["Exact_Match"]["fp"] == 1
    assert results["overall"]["Exact_Match"]["fn"] == 1
    assert results["overall"]["Main_Code_Match"]["tp"] == 2
    assert results["overall"]["Main_Code_Match"]["fp"] == 0
    assert results["overall"]["Main_Code_Match"]["fn"] == 0
